"""Report a rerun_three_engines campaign without pooling historical timings."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    env = json.loads((directory/'environment.json').read_text())
    rows = [json.loads(line) for line in (directory/'observations.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['engine'] != 'comparison_complete']
    grouped = defaultdict(list)
    for row in rows:
        if not row['warmup']:
            grouped[(row['case'], row['engine'])].append(row)
    cells = []
    for job in env['manifest']:
        spec = job['spec']
        from evaluation.workloads import digest
        case = digest(spec)[:16]
        cell = dict(case=case, **spec)
        for engine in ('pyzx', 'cypher', 'hybrid'):
            group = grouped[(case, engine)]
            complete = len(group) == env['repetitions'] and all(r['status'] not in ('error', 'failed') for r in group)
            cell[engine+'_complete'] = complete
            cell[engine+'_active'] = complete and all(r['changed'] for r in group)
            for scope in ('rewrite', 'total'):
                cell[engine+'_'+scope] = stats.median(r[scope+'_s'] for r in group) if complete else None
            if group:
                cell['vertices'] = group[0]['input']['vertices']
                cell['edges'] = group[0]['input']['edges']
        for engine in ('cypher', 'hybrid'):
            for scope in ('rewrite', 'total'):
                a, b = cell['pyzx_'+scope], cell[engine+'_'+scope]
                cell[engine+'_'+scope+'_speedup'] = a/b if a is not None and b and cell['pyzx_active'] and cell[engine+'_active'] else None
        cells.append(cell)
    (directory/'timings.json').write_text(json.dumps(cells, indent=2))
    fields = sorted({k for c in cells for k in c})
    with (directory/'timings.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)
    family_sizes = defaultdict(list)
    for cell in cells:
        family_sizes[(cell['family'], cell['size'])].append(cell)
    summaries = []
    for (family, size), group in sorted(family_sizes.items()):
        summary = dict(family=family, size=size, configurations=len(group))
        for key in ('vertices', 'edges', *[engine+'_'+scope for engine in ('pyzx', 'cypher', 'hybrid') for scope in ('rewrite', 'total')],
                    *[engine+'_'+scope+'_speedup' for engine in ('cypher', 'hybrid') for scope in ('rewrite', 'total')]):
            values = [c.get(key) for c in group]
            summary[key] = stats.geometric_mean(values) if all(v is not None and v > 0 for v in values) else None
        summaries.append(summary)
    (directory/'family-summary.json').write_text(json.dumps(summaries, indent=2))
    fmt = lambda value: '—' if value is None else f'{value:.5g}'
    completion = (directory/'completion.json').exists()
    text = ['# Three-engine benchmark rerun — 2026-10-28', '',
            f"Status: {'COMPLETE' if completion else 'PARTIAL'}. {len(env['manifest'])} planned workload configurations.", '',
            'PyZX 0.10.3 and Memgraph 3.14.0. One warmup and three measured repetitions per engine/configuration. '
            'Each repetition uses the same input and deterministically shuffled engine order. Stock PyZX is timed once and shared by the two database comparisons. '
            'Cypher uses the optimized Python-coordinated Cypher schedule; hybrid additionally uses scoped C++ procedures. '
            'All verification and graph snapshot I/O are outside timers. Rewrite excludes graph import/export; total includes them (PyZX includes graph copying).', '',
            'This is finite-size empirical scaling evidence, not a proof of asymptotic complexity. '
            'Random circuits use depth=20×qubits. Layered circuits use two layers with qubits−1 selected edges per layer. '
            'Rule fans and disjoint copies are different workload shapes. Local-layered seeds generate identical inputs; their repetitions are not independent circuits. '
            'QFT and other adversarial circuits are included in the separate correctness suite, not invented as new scalability families.', '',
            '## Correctness against PyZX', '']
    measured = [r for r in rows if not r['warmup']]
    counts = {}
    for engine in ('pyzx', 'cypher', 'hybrid'):
        group = [r for r in measured if r['engine'] == engine]
        counts[engine] = dict(input_tensor=dict(Counter(r.get('check', {}).get('status', r['status']) for r in group)))
        if engine != 'pyzx':
            for kind in ('tensor', 'graph', 'degree_sequence'):
                counts[engine][kind] = dict(Counter(r.get('against_pyzx', {}).get(kind, {}).get('status', 'missing') for r in group))
        text.append(f'- {engine}: `{json.dumps(counts[engine])}`')
    (directory/'correctness-summary.json').write_text(json.dumps(counts, indent=2))
    text += ['', 'Tensor checks preserve boundary order and compare up to a nonzero global scalar. '
             'Labeled multigraph isomorphism preserves vertex type, phase, wire type/multiplicity and boundary positions, but excludes scalar. '
             'Isomorphism is bounded at 2,048 vertices, 4,096 edges and two seconds. Degree equality is only a weak structural check, not semantic proof. '
             'Different outputs or degree sequences can be equivalent because rewrite normal forms are not unique. '
             'Large tensor/isomorphism limits remain inconclusive. No global correctness claim is made for those outputs.', '',
             '## Largest scale per family', '',
             'Times in seconds; geometric mean of per-seed medians for circuit families. Speedup = PyZX/database; >1 favors the database. '
             'Failed, incomplete and no-op cells receive no speedup.', '',
             '| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher speedup | Hybrid speedup |',
             '|---|---:|---:|---:|---:|---:|---:|']
    largest = [s for s in summaries if s['size'] == max(t['size'] for t in summaries if t['family'] == s['family'])]
    for s in largest:
        text.append('| '+' | '.join([s['family'], str(s['size']), *[fmt(s[k]) for k in ('pyzx_rewrite','cypher_rewrite','hybrid_rewrite','cypher_rewrite_speedup','hybrid_rewrite_speedup')]])+' |')
    text += ['', '## Largest-scale total time', '',
             '| Family | Size | PyZX total | Cypher total | Hybrid total | Cypher speedup | Hybrid speedup |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in largest:
        text.append('| '+' | '.join([s['family'], str(s['size']), *[fmt(s[k]) for k in ('pyzx_total','cypher_total','hybrid_total','cypher_total_speedup','hybrid_total_speedup')]])+' |')
    text += ['', '## Observed growth over the two largest sizes', '',
             'Exponent = log(time ratio)/log(size ratio), based on rewrite medians aggregated across seeds. '
             'These local slopes are sensitive to fixed overhead, matching order and graph density; they are not Big-O estimates.', '',
             '| Family | Sizes | PyZX exponent | Cypher exponent | Hybrid exponent |',
             '|---|---|---:|---:|---:|']
    for family in sorted({s['family'] for s in summaries}):
        a, b = sorted([s for s in summaries if s['family'] == family], key=lambda s:s['size'])[-2:]
        exponents = []
        for engine in ('pyzx', 'cypher', 'hybrid'):
            x, y = a[engine+'_rewrite'], b[engine+'_rewrite']
            exponents.append(math.log(y/x)/math.log(b['size']/a['size']) if x and y else None)
        text.append('| '+' | '.join([family, f"{a['size']}→{b['size']}", *map(fmt, exponents)])+' |')
    text += ['', '## All scales', '',
             '| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | PyZX total | Cypher total | Hybrid total |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        text.append('| '+' | '.join([s['family'], str(s['size']), *[fmt(s[e+'_'+t]) for t in ('rewrite','total') for e in ('pyzx','cypher','hybrid')]])+' |')
    text += ['', 'Individual seed/configuration medians: [timings.csv](timings.csv). '
             'All trials, checks and timings: [observations.jsonl](observations.jsonl). '
             'Exact source archive: [sources.zip](sources.zip). '
             'Regressions and all 223 correctness fixtures: [validation report](../validation/REPORT.md).', '', '## Errors', '']
    for row in rows:
        if row['status'] in ('error','failed'):
            text.append(f"- {row['spec']} {row['engine']} rep={row['repetition']}: {row.get('error', row.get('check'))}")
    (directory/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
