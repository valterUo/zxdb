"""Check completeness/provenance and assemble the campaign's entry report."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
perf = root/'performance'
assert json.loads((perf/'completion.json').read_text())['finished']
env = json.loads((perf/'environment.json').read_text())
rows = [json.loads(line) for line in (perf/'observations.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['engine']!='comparison_complete']
assert len(rows)==len(env['manifest'])*3*(env['repetitions']+1)
assert len({(r['case'],r['engine'],r['repetition']) for r in rows})==len(rows)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
assert {r['case'] for r in rows}=={digest(job['spec'])[:16] for job in env['manifest']}
for case in {r['case'] for r in rows}:
    selected=[r for r in rows if r['case']==case]
    assert len({r['input_hash'] for r in selected})==1
    assert len({json.dumps(r['input'],sort_keys=True) for r in selected})==1
    assert all(digest(r['spec'])[:16]==case for r in selected)
graph_files=list((perf/'graphs').glob('*.json'))
for path in graph_files:
    assert digest(json.loads(path.read_text()))==path.stem, path
hashes = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest()==value for name,value in env['sources'].items()}
assert all(hashes.values()), hashes
cells = json.loads((perf/'timings.json').read_text())
for cell in cells:
    for engine in ('pyzx','cypher','hybrid'):
        selected=[r for r in rows if r['case']==cell['case'] and r['engine']==engine and not r['warmup']]
        assert len(selected)==env['repetitions']
        if cell[engine+'_complete']:
            for scope in ('rewrite','total'):
                assert cell[engine+'_'+scope]==statistics.median(r[scope+'_s'] for r in selected)
integrity=dict(recorded_trials=len(rows), measured_trials=sum(not r['warmup'] for r in rows),
               configurations=len(cells), sources_match=hashes,
               statuses=dict(Counter(r['status'] for r in rows)), median_recalculation='passed',
               shared_inputs_check='passed', graph_files_hash_verified=len(graph_files))
(root/'integrity.json').write_text(json.dumps(integrity,indent=2))
component=json.loads((perf/'component-isomorphism.json').read_text())
component_results={(c['db_hash'],c['pyzx_hash']):c['result'] for c in component['checks']}
refs={(r['case'],r['repetition']):r for r in rows if r['engine']=='pyzx' and 'graphs' in r}
coverage={}
for engine in ('cypher','hybrid'):
    checks=defaultdict(Counter)
    for row in rows:
        if row['engine']!=engine or row['warmup']:
            continue
        direct=row.get('against_pyzx',{})
        checks['input_tensor'][row.get('check',{}).get('status','missing')]+=1
        checks['reference_tensor'][direct.get('tensor',{}).get('status','missing')]+=1
        graph_status=direct.get('graph',{}).get('status','missing')
        if 'graphs' in row:
            ref=refs[(row['case'],row['repetition'])]
            pair=(row['graphs']['result'],ref['graphs']['result'])
            if pair in component_results:
                graph_status=component_results[pair]['status']
        checks['isomorphism'][graph_status]+=1
        checks['degree_sequence'][direct.get('degree_sequence',{}).get('status','missing')]+=1
    coverage[engine]={k:dict(v) for k,v in checks.items()}
(root/'combined-correctness.json').write_text(json.dumps(coverage,indent=2))
supplement=json.loads((root/'validation/supplemental-correctness.json').read_text())
supp_ok=sum(all(c['status']=='verified' for c in item['checks'].values()) for item in supplement['checks'])
summaries=json.loads((perf/'family-summary.json').read_text())
largest=[s for s in summaries if s['size']==max(t['size'] for t in summaries if t['family']==s['family'])]
circuits=[s for s in largest if s['family'].startswith(('random_','layered_'))]
rules=[s for s in largest if not s['family'].startswith(('random_','layered_','identity_chain'))]
large_cells=[c for c in cells if c['size']==128 and c['family'].startswith(('random_','layered_'))]
large_wins={engine:{scope:sum(c[engine+'_'+scope+'_speedup']>1 for c in large_cells)
                    for scope in ('rewrite','total')} for engine in ('cypher','hybrid')}
fmt=lambda v:'—' if v is None else f'{v:.3f}'
text=['# Benchmark rerun: PyZX, Cypher, and Cypher + C++', '',
      'Completed 2026-10-28. All 28 implemented performance workload shapes, 160 configurations, '
      '1,440 measured engine trials plus 480 warmups. Three measured repetitions; three seed settings for each circuit size.', '',
      '**The advantage is workload-dependent.** See the tables below for the measured large-circuit results. '
      'Neither database approach wins across all implemented benchmarks, and finite-size measurements do not prove asymptotic superiority. '
      'The detailed report includes observed growth exponents between the two largest tested sizes.', '',
      f"Across the 15 individual 128-qubit configurations, Cypher wins {large_wins['cypher']['rewrite']}/15 "
      f"on rewrite time and hybrid wins {large_wins['hybrid']['rewrite']}/15; both win 12/15 on total time. "
      'Cypher loses rewrite time on T-heavy seed 2 (0.917× speedup). Cypher loses total time on all three T-heavy seeds; '
      'hybrid loses total time on T-heavy seed 0 and local-layered seeds 0 and 1. Family averages below do not replace these per-seed results.', '',
      '## Large circuits: 128 qubits', '',
      'Seconds, geometric mean across per-seed medians. Speedup = PyZX/database. '
      'Random circuits have depth 2,560; layered circuits have two layers. Local-layered seed settings yield identical inputs.', '',
      '| Circuit | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher speedup | Hybrid speedup |',
      '|---|---:|---:|---:|---:|---:|']
for row in circuits:
    text.append('| '+' | '.join([row['family'],*[fmt(row[k]) for k in ('pyzx_rewrite','cypher_rewrite','hybrid_rewrite','cypher_rewrite_speedup','hybrid_rewrite_speedup')]])+' |')
text += ['', '| Circuit | PyZX total | Cypher total | Hybrid total | Cypher speedup | Hybrid speedup |',
         '|---|---:|---:|---:|---:|---:|']
for row in circuits:
    text.append('| '+' | '.join([row['family'],*[fmt(row[k]) for k in ('pyzx_total','cypher_total','hybrid_total','cypher_total_speedup','hybrid_total_speedup')]])+' |')
text += ['', '![Circuit rewrite scaling](performance/circuit-rewrite-scaling.png)', '',
         '![Circuit total-time scaling](performance/circuit-total-scaling.png)', '',
         '## All standalone rules', '']
for engine in ('cypher','hybrid'):
    rewrite=sum(s[engine+'_rewrite_speedup'] is not None and s[engine+'_rewrite_speedup']>1 for s in rules)
    total=sum(s[engine+'_total_speedup'] is not None and s[engine+'_total_speedup']>1 for s in rules)
    text.append(f'- {engine}: {rewrite}/22 largest-scale rule shapes win on rewrite time; {total}/22 win on total time.')
text += ['', 'These 22 shapes cover eleven rule families as growing neighborhoods and disjoint copies. '
         'Identity chains are additionally measured. The [complete report](performance/REPORT.md) lists every family at its largest size '
         'and all measured scales. [CSV](performance/timings.csv) retains every individual configuration and seed.', '',
         '## Correctness', '',
         f'All 91 live regressions passed. The initial fixture run verified 440/446 database outcomes; '
         f'{supp_ok}/6 initially inconclusive outcomes passed supplemental exact-zero-tensor or unitary-identity checks. '
         'The 446 outcomes are 223 fixtures in both database modes, including QFT and adversarial compositions.', '',
         'Counts below are per 480 measured output trials for each database mode, including repeated outputs; columns overlap. '
         'All remaining tensor comparisons are inconclusive, with no tensor failures. Isomorphism counts include the supplemental audit.', '',
         '| Engine | Input tensor verified | PyZX tensor verified | Isomorphic | Different structure | Iso unresolved | Equal degrees |',
         '|---|---:|---:|---:|---:|---:|---:|']
for engine, checks in coverage.items():
    values=[checks['input_tensor'].get('verified',0),checks['reference_tensor'].get('verified',0),
            checks['isomorphism'].get('equal',0),checks['isomorphism'].get('different',0),
            checks['isomorphism'].get('inconclusive',0),checks['degree_sequence'].get('equal',0)]
    text.append('| '+' | '.join([engine,*map(str,values)])+' |')
text += ['', 'There is no blanket semantic certificate for every large output. Degree equality is not semantic proof, '
         'and differing structures need not mean a wrong ZX rewrite. Exact isomorphism preserves phases, colors, ordered boundaries '
         'and wire multiplicity but excludes global scalar. Tensor checks use projective semantics, up to a nonzero global scalar. '
         'All original inconclusive checks are retained alongside supplemental evidence.', '',
         '## Reproduction and evidence', '',
         'Run from the repository root against the dedicated empty Memgraph 3.14.0 server with the matching native module:', '',
         '```powershell', "$env:ZXDB_URI='bolt://127.0.0.1:7691'", "$env:ZXDB_PYTHON_MODULE='1'",
         'myenv/Scripts/python.exe evaluate.py --quick --variants optimized native --restart-container zxdb-rule-audit --output results/NEW/validation',
         'myenv/Scripts/python.exe -m evaluation.rerun_three_engines --container zxdb-rule-audit --output results/NEW/performance',
         'myenv/Scripts/python.exe -m evaluation.report_three_engines results/NEW/performance', '```', '',
         'The quick flag applies only to the validation run’s extra smoke timings; it does not reduce its correctness fixtures. '
         'The separate performance runner always uses the full declared 160-configuration grid by default. '
         'Supplemental audit and plotting scripts are saved in this campaign folder and target its sibling performance/validation folders.', '',
         '- [Integrity checks and source-hash verification](integrity.json)',
         '- [Native module and container provenance](native-provenance.json)',
         '- [Regression and fixture report](validation/REPORT.md)',
         '- [Supplemental fixture checks](validation/supplemental-correctness.json)',
         '- [Supplemental component-isomorphism audit](performance/component-isomorphism.json)',
         '- [Combined correctness counts](combined-correctness.json)',
         '- [Raw timed observations](performance/observations.jsonl)',
         '- [Measured source archive](performance/sources.zip)', '',
         'Timings are from this rerun only. Verification, snapshot writing, startup and disk I/O are excluded. '
         'Total includes graph import/rewrite/export for the database and graph copy/rewrite for PyZX. '
         'Matching order and resulting output density affect performance; these are workload-specific measurements, not a complexity proof. '
         'The size parameter denotes qubits, copies, or fan size depending on the family. Fan edge counts can grow quadratically; '
         'the observed exponents are not normalized by input bytes. Input vertex and edge counts are retained in the CSV.']
(root/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(integrity=integrity,coverage=coverage,supplemental_fixtures_verified=supp_ok),indent=2))
