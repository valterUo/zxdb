"""Merge standard and extended measurements into a three-engine audit report."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def geometric(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    folder = args.campaign.resolve()
    completion = folder/'comparison/completion.json'
    finished = completion.exists() and json.loads(completion.read_text()).get('finished',False)
    standard = [json.loads(line) for line in (folder/'current-standard/observations.jsonl').read_text().splitlines()]
    extended = [json.loads(line) for line in (folder/'comparison/observations.jsonl').read_text().splitlines()]
    samples = defaultdict(list)
    specs = {}
    input_hashes = {}
    for r in standard:
        if r['stage'] != 'benchmark':
            continue
        case = r['case']
        specs[case] = r['spec']
        input_hashes[case] = r['input_hash']
        engine = 'cypher' if r['variant'] == 'optimized' else 'hybrid'
        for target, prefix in ((engine, 'db'), ('pyzx', 'pyzx')):
            samples[case, target].append(dict(
                warmup=r['warmup'], status=r['status'], changed=r.get(prefix+'_changed'),
                rewrite_s=r.get(prefix+'_rewrite_s'), total_s=r.get(prefix+'_total_s'),
                queries=r.get('queries') if prefix=='db' else None,
                output=r.get(prefix+'_output',{})))
    for r in extended:
        case = r['case']
        specs[case] = r['spec']
        if case in input_hashes and input_hashes[case] != r['input_hash']:
            raise ValueError('Input mismatch for '+case)
        input_hashes[case] = r['input_hash']
        samples[case, r['engine']].append(r)
    engines = ('pyzx', 'cypher', 'hybrid', 'before_cypher', 'before_hybrid')
    rows = []
    for case, spec in sorted(specs.items(), key=lambda x: (x[1]['family'], x[1]['size'], x[1].get('depth',0),x[1]['seed'])):
        row = dict(case=case, **spec, input_hash=input_hashes[case])
        for engine in engines:
            group = samples[case, engine]
            measured = [r for r in group if not r['warmup']]
            # Standard evaluator runs PyZX once for each DB variant: six
            # measured samples. Extended comparison shares three samples.
            expected = 6 if engine == 'pyzx' and any(
                r['stage'] == 'benchmark' and r['case'] == case for r in standard) else 3
            complete = len(measured) == expected and all(r['status'] in ('verified','inconclusive') for r in group)
            row[engine+'_complete'] = complete
            row[engine+'_samples'] = len(measured)
            row[engine+'_statuses'] = dict(Counter(r['status'] for r in measured))
            row[engine+'_active'] = complete and all(r.get('changed') for r in measured)
            query_counts = [r.get('queries') for r in measured]
            row[engine+'_queries'] = statistics.median(query_counts) if complete and all(
                n is not None for n in query_counts) else None
            for metric in ('vertices','edges'):
                values = [r.get('output',{}).get(metric) for r in measured]
                row[engine+'_output_'+metric] = statistics.median(values) if complete and all(
                    n is not None for n in values) else None
            for scope in ('rewrite','total'):
                times = [r[scope+'_s'] for r in measured] if complete else []
                row[engine+'_'+scope+'_s'] = statistics.median(times) if times else None
                row[engine+'_'+scope+'_range'] = [min(times),max(times)] if times else None
        for engine in ('cypher','hybrid'):
            for scope in ('rewrite','total'):
                current = row[engine+'_'+scope+'_s']
                old = row['before_'+engine+'_'+scope+'_s']
                pyzx = row['pyzx_'+scope+'_s']
                row[engine+'_'+scope+'_change_pct'] = 100*(current/old-1) if current and old else None
                row[engine+'_'+scope+'_speedup'] = pyzx/current if current and pyzx and row[engine+'_active'] and row['pyzx_active'] else None
        rows.append(row)
    (folder/'timings.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields = ['case','family','size','seed','depth','input_hash']
    fields += [key for key in rows[0] if key not in fields]
    with (folder/'timings.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    # Historical comparisons require both matching configuration AND matching
    # input hashes, not merely a shared family label.
    repo = Path(__file__).resolve().parents[1]
    historical_paths = [repo/'results/cypher-optimization/neighborhood-cost-repeated',
                        repo.parent/'zxdb-backup-20261026/validation-history-20261027/native',
                        repo.parent/'zxdb-backup-20261026/validation-history-20261027/native-128-stable',
                        repo.parent/'zxdb-backup-20261026/validation-history-20261027/full']
    historical = []
    by_case = {r['case']:r for r in rows}
    for path in historical_paths:
        if not (path/'summary.json').exists() or not (path/'observations.jsonl').exists():
            continue
        old_environment = json.loads((path/'environment.json').read_text())
        old_rows = [json.loads(s) for s in (path/'observations.jsonl').read_text().splitlines()]
        for old in json.loads((path/'summary.json').read_text()):
            if old['case'] not in by_case or old['variant'] not in ('optimized','native'):
                continue
            current = by_case[old['case']]
            hashes = {r['input_hash'] for r in old_rows if r['stage']=='benchmark' and r['case']==old['case'] and r['variant']==old['variant']}
            if hashes != {current['input_hash']}:
                continue
            engine = 'hybrid' if old['variant']=='native' else 'cypher'
            h = dict(source=str(path), case=old['case'], spec=old['spec'], engine=engine,
                     historical_complete=old['complete'], current_complete=current[engine+'_complete'],
                     historical_database_version=old_environment.get('database_version'),
                     historical_versions=old_environment.get('versions'),
                     historical_rewrite_hashes={k:v for k,v in old_environment.get('source_hashes',{}).items()
                         if k.replace('\\','/') in ('zxdb/zxdb.py','zxdb/main_queries.json','zxdb_qm/zxqmcpp.cpp')})
            for scope in ('rewrite','total'):
                a,b = old.get('db_'+scope+'_s'),current[engine+'_'+scope+'_s']
                h.update({scope+'_old_s':a,scope+'_new_s':b,
                          scope+'_change_pct':100*(b/a-1) if a and b and old['complete']
                          and current[engine+'_complete'] else None})
            historical.append(h)
    (folder/'historical-comparison.json').write_text(json.dumps(historical,indent=2)+'\n')
    def number(x):
        return '—' if x is None else f'{x:.6g}'
    def percent(x):
        return '—' if x is None else f'{x:+.1f}%'
    groups = defaultdict(list)
    for r in rows:
        groups[r['family'],r['size'],r.get('depth')].append(r)
    aggregates = []
    for (family,size,depth), group in sorted(groups.items()):
        a = dict(family=family,size=size,depth=depth,seeds=len(group))
        for engine in engines:
            for scope in ('rewrite','total'):
                values = [r[engine+'_'+scope+'_s'] for r in group]
                a[engine+'_'+scope+'_s'] = geometric(values) if all(values) else None
        for engine in ('cypher','hybrid'):
            for scope in ('rewrite','total'):
                old,new = a['before_'+engine+'_'+scope+'_s'],a[engine+'_'+scope+'_s']
                a[engine+'_'+scope+'_change_pct'] = 100*(new/old-1) if old and new else None
        aggregates.append(a)
    (folder/'family-summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    largest = [a for a in aggregates if a['size'] == max(b['size'] for b in aggregates if b['family']==a['family'])]
    text = ['# Post-audit performance comparison', '',
            'Run status: '+('finished.' if finished else 'PARTIAL — extended measurements are still running.'), '',
            f'{len(rows)} workload configurations, all eleven rule families, five circuit families, and identity chains.', '',
            'Times are seconds. Each database cell has one warmup and three measured repetitions. '
            'The standard runner supplies two PyZX groups, each with one warmup and three measured '
            'samples; the six measured samples are pooled here. Large-circuit PyZX has one warmup '
            'and three measured samples, shared across database modes. '
            'Speedups in this report are ratios of median times. Positive before/after change means slower.', '',
            'Cypher means the optimized Cypher-only schedule. Hybrid uses scoped C++ procedures for '
            'pivot, local complementation, gadget/boundary pivot, and supplementarity; other rules remain Cypher. '
            'The C++ source is unchanged by the audit. Before-fix database paths use the saved pre-audit '
            'Python and query sources on the same server and installed dependencies. '
            '[Source provenance](baseline-provenance.json) confirms these three rewrite sources match '
            'the previous completed Cypher campaign after line-ending normalization.', '',
            '## Correctness and completeness', '',
            str(dict(Counter((r['stage']+':'+r['status']) for r in standard))), '',
            'Extended observations: '+str(dict(Counter(r['status'] for r in extended)))+'.', '',
            'Inconclusive tensor checks are not semantic successes: reasons include tensor resource bounds '
            'and numerically ambiguous near-zero contractions. '
            'Failed or incomplete cells have no reported median or speedup. No-op cases retain elapsed times '
            'but receive no speedup. Timings exclude verification and snapshot writing.', '',
            '## Largest tested size in each family', '',
            'Circuit rows use geometric means across seed medians (three seeds in the completed grid). Other rows have one '
            'deterministic input. These aggregates do not hide the individual rows below.', '',
            '| Workload | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher change | Hybrid change |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in largest:
        text.append(f"| {r['family']} | {r['size']} | {number(r['pyzx_rewrite_s'])} | "
                    f"{number(r['cypher_rewrite_s'])} | {number(r['hybrid_rewrite_s'])} | "
                    f"{percent(r['cypher_rewrite_change_pct'])} | {percent(r['hybrid_rewrite_change_pct'])} |")
    text += ['', '## All workload timings', '',
            '| Workload | Size | Depth | Seed | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher change vs pre-fix | Hybrid change vs pre-fix |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['family']} | {r['size']} | {r.get('depth','—')} | {r['seed']} | "
                    f"{number(r['pyzx_rewrite_s'])} | {number(r['cypher_rewrite_s'])} | "
                    f"{number(r['hybrid_rewrite_s'])} | {percent(r['cypher_rewrite_change_pct'])} | "
                    f"{percent(r['hybrid_rewrite_change_pct'])} |")
    text += ['', '## Interpretation and evidence', '',
             'The before/after percentages are observed timing differences, not causal guarantees or '
             'statistical significance tests. Standard current-code timings precede the pre-fix rerun; '
             'large configurations alternate all paths in a deterministic shuffled order. '
             'Frequency scaling, thermal state and matching order can affect results. Use the per-cell '
             'ranges and graph metrics in the raw records when interpreting changes.', '',
             '[Full timings, including total time, ranges and verification status](timings.csv) · '
             '[Machine-readable timings](timings.json) · '
             '[Total-time table](TOTAL_TIMES.md) · '
             '[Historical comparisons with matching input hashes](historical-comparison.json) · '
             '[Standard run](current-standard/REPORT.md) · '
             '[Extended manifest and source hashes](comparison/environment.json)', '',
             'Historical comparisons use the same workload specification and exact input hash. They can '
             'involve different source revisions or run conditions, so the contemporaneous pre-fix rerun '
             'is the stronger comparison for the audit changes. Neither comparison reruns old broken '
             'code on known invalid audit counterexamples.']
    (folder/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    total = ['# Total times: copy or import/rewrite/export', '',
             'Seconds; medians of measured repetitions. Database startup and disk I/O are excluded. '
             'PyZX total includes copying the input graph.', '',
             '| Workload | Size | Depth | Seed | PyZX | Cypher | Hybrid | Cypher change | Hybrid change |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        total.append(f"| {r['family']} | {r['size']} | {r.get('depth','—')} | {r['seed']} | "
                     f"{number(r['pyzx_total_s'])} | {number(r['cypher_total_s'])} | {number(r['hybrid_total_s'])} | "
                     f"{percent(r['cypher_total_change_pct'])} | {percent(r['hybrid_total_change_pct'])} |")
    (folder/'TOTAL_TIMES.md').write_text('\n'.join(total)+'\n',encoding='utf-8')
    large = [r for r in aggregates if r['size']==128 and r['family'].startswith(('random_','layered_'))]
    if len(large)==5 and all(r['seeds']==3 for r in large) and all(
            r[e+'_'+s+'_s'] for r in large for e in engines[:3] for s in ('rewrite','total')):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels = {'random_clifford':'Clifford','random_clifford_t':'Clifford+T',
                  'random_t_heavy':'T-heavy','layered_local':'Local layered','layered_global':'Global layered'}
        fig, axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for ax, scope in zip(axes, ('rewrite','total')):
            for offset,(engine,label,color) in enumerate(zip(engines[:3],
                    ('PyZX','Cypher only','Cypher + C++'),('#748099','#3478c0','#27936b'))):
                ax.bar([i+(offset-1)*.25 for i in range(5)],
                       [r[engine+'_'+scope+'_s'] for r in large],width=.24,label=label,color=color)
            ax.set_yscale('log')
            ax.set_xticks(range(5),[labels[r['family']] for r in large],rotation=20,ha='right')
            ax.set_ylabel('Seconds (log scale)')
            ax.set_title('Rewrite time' if scope=='rewrite' else 'Total time')
            ax.grid(axis='y',alpha=.2)
        axes[0].legend()
        fig.suptitle('128-qubit benchmarks after the audit\nGeometric mean across three seed medians; three measured repetitions')
        fig.savefig(folder/'large-circuits.png',dpi=180)
        plt.close(fig)
    print(f'Wrote {len(rows)} workload rows and {len(historical)} historical matches.')


if __name__ == '__main__':
    main()
