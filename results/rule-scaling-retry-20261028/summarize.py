"""Summarize fresh retries separately; earlier attempts are never overwritten."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ENGINES = ('pyzx', 'cypher', 'hybrid')


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    progress = json.loads((ROOT / 'progress.json').read_text()) if (ROOT / 'progress.json').exists() else []
    cells, errors, checks = [], [], {e: Counter() for e in ENGINES}
    for path in sorted(ROOT.glob('*/observations.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        trials = [r for r in rows if r['engine'] in ENGINES]
        if not trials:
            continue
        cell = dict(case=trials[0]['case'], **trials[0]['spec'],
                    vertices=trials[0]['input']['vertices'], edges=trials[0]['input']['edges'])
        for engine in ENGINES:
            measured = [r for r in trials if r['engine'] == engine and not r['warmup']]
            complete = len(measured) == 3 and {r['repetition'] for r in measured} == {1, 2, 3}
            complete = complete and all(r['status'] not in ('error', 'failed') for r in measured)
            cell[engine + '_complete'] = complete
            cell[engine + '_active'] = complete and all(r.get('changed') for r in measured)
            checks[engine].update(r.get('check', {}).get('status', r['status']) for r in measured)
            for scope in ('rewrite', 'total'):
                cell[engine + '_' + scope] = median(r[scope + '_s'] for r in measured) if complete else None
        for engine in ENGINES[1:]:
            for scope in ('rewrite', 'total'):
                a, b = cell['pyzx_' + scope], cell[engine + '_' + scope]
                cell[engine + '_' + scope + '_speedup'] = (
                    a / b if a and b and cell['pyzx_active'] and cell[engine + '_active'] else None)
        cells.append(cell)
        errors += [dict(directory=path.parent.name, **r) for r in trials if r['status'] in ('error', 'failed')]
    (ROOT / 'timings.json').write_text(json.dumps(cells, indent=2))
    if cells:
        with (ROOT / 'timings.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for c in cells for k in c}))
            writer.writeheader()
            writer.writerows(cells)
    (ROOT / 'correctness-summary.json').write_text(json.dumps(checks, indent=2))
    counts = dict(Counter(p['status'] for p in progress))
    text = ['# Larger-rule retries - 2026-10-28', '',
            f'Status: {counts}; {len(progress)}/{len(plan["jobs"])} retry cases finished.', '',
            'Fresh retries of the six failed or interrupted configurations in '
            '[the previous extension](../rule-scaling-20261028/REPORT.md). '
            'Original failures and timings remain untouched. No repetition is pooled across attempts.', '',
            f'Per-engine database I/O/rewrite limit: {plan["engine_timeout_seconds"]} seconds. '
            f'Per-case process limit: {plan["case_timeout_seconds"]} seconds. '
            'No shared campaign cutoff and no projected-runtime early stops. '
            'One warmup and three measured repetitions per engine. '
            'The dedicated database is restarted before each retry; its memory limit is unchanged. '
            'Queries, rewrite implementations, workload generators, and the native module are unchanged.', '',
            'This server resets query.timeout to 600 seconds on restart. The runner now raises a '
            'smaller nonzero server timeout to the requested engine limit before timing; environment.json '
            'records both values. Earlier retries ran below their recorded 600-second server cap. '
            'The current Hadamard case received a runtime override recorded in server-timeout-override.json. '
            'The original 600-second setting is restored after the campaign. For reproduction, restore '
            'the dedicated server\'s prior query.timeout value afterward.', '',
            'Seconds below are medians of complete measured triplets. Speedup is PyZX/database; '
            'above one favors the database. Missing or failed triplets have no median or speedup. '
            'Total includes graph import/export (PyZX: copying), but excludes startup and verification. '
            'Timing completion is distinct from semantic verification; large tensor checks can remain inconclusive.', '',
            '| Workload | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher rewrite speedup | Hybrid rewrite speedup | Cypher total speedup | Hybrid total speedup |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['pyzx_rewrite', 'cypher_rewrite', 'hybrid_rewrite', 'cypher_rewrite_speedup',
            'hybrid_rewrite_speedup', 'cypher_total_speedup', 'hybrid_total_speedup']
    for c in cells:
        text.append('| ' + ' | '.join([c['family'], str(c['size']),
            *['--' if c[k] is None else f'{c[k]:.6g}' for k in keys]]) + ' |')
    text += ['', '## Retry outcomes', '']
    text += [f'- {p["spec"]["family"]}, n={p["spec"]["size"]}: {p["status"]} '
             f'({p.get("wall_seconds", 0):.1f} seconds).' for p in progress]
    text += ['', '## Errors, including warmups', '']
    text += [f'- {r["directory"]}, {r["engine"]}, repetition {r["repetition"]}: '
             f'{r.get("error", r.get("check"))}' for r in errors]
    for path in ROOT.glob('*/container-state-at-failure.json'):
        if path.read_text().strip():
            state = json.loads(path.read_text())
            text.append(f'- {path.parent.name}: container exit code {state["ExitCode"]}, '
                        f'OOMKilled={state["OOMKilled"]}.')
    text += ['', '## Reproduction', '', '```sh',
             'python -m evaluation.extend_rule_scaling --retry-from results/rule-scaling-20261028 '
             '--output results/new-retry --budget 0 --case-timeout 10800 --engine-timeout 1800',
             'python results/rule-scaling-retry-20261028/summarize.py', '```', '',
             'Each case retains its source archive, environment, input/output graphs, paired '
             'observations and immediate per-engine observations. The immediate stream preserves '
             'evidence if a later engine crashes; the summary uses only the paired observations.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print(counts)
    for c in cells:
        print(c['family'], c['size'], {k: c[k] for k in keys})


if __name__ == '__main__':
    main()
