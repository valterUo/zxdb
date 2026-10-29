"""Rebuild summaries from recorded trials; never pool with the earlier campaign."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ENGINES = ('pyzx', 'cypher', 'hybrid')


def main():
    cells, observations = [], []
    for path in sorted(ROOT.glob('*/observations.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        measured = [r for r in rows if r['engine'] in ENGINES and not r['warmup']]
        observations.extend(rows)
        if not measured:
            continue
        cell = dict(case=measured[0]['case'], **measured[0]['spec'],
                    vertices=measured[0]['input']['vertices'], edges=measured[0]['input']['edges'])
        for engine in ENGINES:
            group = [r for r in measured if r['engine'] == engine]
            complete = len(group) == 3 and all(r['status'] not in ('error', 'failed') for r in group)
            cell[engine + '_complete'] = complete
            cell[engine + '_active'] = complete and all(r.get('changed') for r in group)
            for scope in ('rewrite', 'total'):
                cell[engine + '_' + scope] = median(r[scope + '_s'] for r in group) if complete else None
        for engine in ENGINES[1:]:
            for scope in ('rewrite', 'total'):
                a, b = cell['pyzx_' + scope], cell[engine + '_' + scope]
                cell[engine + '_' + scope + '_speedup'] = (
                    a / b if a and b and cell['pyzx_active'] and cell[engine + '_active'] else None)
        cells.append(cell)
    if not cells:
        return
    (ROOT / 'timings.json').write_text(json.dumps(cells, indent=2))
    with (ROOT / 'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for c in cells for k in c}))
        writer.writeheader()
        writer.writerows(cells)
    progress = json.loads((ROOT / 'progress.json').read_text())
    plan = json.loads((ROOT / 'plan.json').read_text())
    counts = Counter(p['status'] for p in progress)
    checks = {e: dict(Counter(r.get('check', {}).get('status', r['status']) for r in observations
                             if r['engine'] == e and not r['warmup'])) for e in ENGINES}
    (ROOT / 'correctness-summary.json').write_text(json.dumps(checks, indent=2))
    text = ['# Larger standalone-rule timings - 2026-10-28', '',
            f'Campaign status: {dict(counts)}; {len(progress)}/{len(plan["jobs"])} jobs accounted for.', '',
            'Three measured repetitions after one warmup for each completed case. '
            'PyZX, Cypher, and hybrid share the same input, with deterministically shuffled engine order. '
            'All timings below are medians. Old campaign measurements are not pooled.', '',
            'Queries, rewrite implementations, native module and graph generators are unchanged. '
            'Only benchmark sizes increase. Dense fan sizes are capped at 1024 because outputs can have '
            'quadratically many edges. Other workloads are planned up to size 16384. '
            'A one-hour campaign budget and 1200-second case deadline bound resource use. '
            'The final case deadline is shortened to the remaining campaign budget; '
            'a case_timeout is not an engine-specific 1200-second runtime lower bound. '
            'A failure stops larger sizes of that workload family, regardless of which engine failed. '
            'No stopping criterion depends on a database win. During this run, resource-based early '
            'stops were also applied when warmup duration projected beyond the case budget; '
            'resource-stop.json files record those decisions. These manual early stops are not '
            'automatically reproduced by the command below, so the reached subset may differ.', '',
            'Total includes in-memory graph import/export, or PyZX graph copying, but excludes '
            'startup, graph snapshots and verification. A timed completion is not a semantic certificate. '
            'Large tensor checks remain inconclusive; see correctness-summary.json and per-trial checks. '
            'Missing, failed or incomplete measurements are not reported as wins.', '',
            '| Workload | Size n | Vertices | PyZX rewrite (s) | Cypher rewrite (s) | Hybrid rewrite (s) | Cypher rewrite speedup | Hybrid rewrite speedup | Cypher total speedup | Hybrid total speedup |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['pyzx_rewrite', 'cypher_rewrite', 'hybrid_rewrite', 'cypher_rewrite_speedup',
            'hybrid_rewrite_speedup', 'cypher_total_speedup', 'hybrid_total_speedup']
    for c in sorted(cells, key=lambda c: (c['family'], c['size'])):
        values = ['--' if c[k] is None else f'{c[k]:.5g}' for k in keys]
        text.append('| ' + ' | '.join([c['family'], str(c['size']), str(c['vertices']), *values]) + ' |')
    text += ['', 'Speedup = PyZX/database; above 1 favors the database. '
             '`copies_*` measures disjoint size-2 motifs. Other family sizes retain their existing '
             'generator meaning; fan size is not a qubit count.', '', '## Uncompleted configurations', '']
    text += [f'- {p["spec"]["family"]}, n={p["spec"]["size"]}: {p["status"]}'
             for p in progress if p['status'] != 'complete']
    text += ['', '## Recorded errors (including warmups)', '']
    text += [f'- {r["spec"]["family"]}, n={r["spec"]["size"]}, {r["engine"]}, '
             f'repetition {r["repetition"]}: {r.get("error", r.get("check"))}'
             for r in observations if r['engine'] in ENGINES and r['status'] in ('error', 'failed')]
    for path in ROOT.glob('*/container-state-at-failure.json'):
        state = json.loads(path.read_text(encoding='utf-8-sig'))
        text.append(f'- {path.parent.name}: container exit code {state["ExitCode"]}; '
                    f'Docker OOMKilled={state["OOMKilled"]}. See the preserved container log.')
    for path in ROOT.glob('*/resource-stop.json'):
        decision = json.loads(path.read_text())
        text.append(f'- {path.parent.name}: early resource stop. {decision["reason"]}')
    text += ['', '## Figures and validation', '',
             '[Updated four-page TikZ figure preview](figures/main.pdf). '
             '[Combined plot inputs](combined-timings.csv) include each row\'s source campaign. '
             'Each configuration is taken from exactly one campaign; repetitions are not pooled. '
             'The figures include earlier smaller sizes and completed larger sizes. '
             'Missing larger cases remain listed above.', '',
             '[Summary validation](summary-validation.json) checks every reported median and ratio '
             'against the raw trials. [Source integrity](integrity.json) and '
             '[comparison to the earlier source](baseline-source-comparison.json) record unchanged implementations.', '',
             '## Reproduction', '', '```sh',
             'python -m evaluation.extend_rule_scaling --output results/new-rule-scaling --budget 3600',
             'python results/rule-scaling-20261028/summarize.py',
             'python results/rule-scaling-20261028/make_figures.py', '```', '',
             'The first command requires the existing dedicated native-enabled database on port 7691. '
             'The second rebuilds this saved campaign summary. Each case directory retains its '
             'manifest, source archive, environment, graph snapshots, trial observations and run log. '
             'plan.json records the full grid and progress.json records resource-limited omissions.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print(dict(counts), 'timing rows:', len(cells))
    for c in cells:
        if c['family'] in ('spider_fusion', 'identity', 'copies_spider_fusion', 'copies_identity'):
            print(c['family'], c['size'], {k: c[k] for k in keys[3:]})


if __name__ == '__main__':
    main()
