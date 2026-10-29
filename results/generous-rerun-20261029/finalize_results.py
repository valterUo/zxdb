"""Freeze and summarize the stopped campaign without pooling historical runs."""
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as stats
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT/'finalized'
ENGINES = ('pyzx', 'cypher', 'hybrid')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def table(headers, records):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
            *['| '+' | '.join(str(x) for x in row)+' |' for row in records], '']


def fmt(n):
    return '—' if n is None else f'{n:.3g}'


def main():
    OUT.mkdir(exist_ok=False)
    plan, state = read(ROOT/'plan.json'), read(ROOT/'progress.json')
    assert state['status'] == 'finalized'
    completed = {c['index']: c for c in state['completed']}
    evidence, cells, outcomes = [], [], []
    for index, job in enumerate(plan['jobs']):
        status = completed.get(index)
        case = ROOT/status['directory'] if status and status.get('directory') else None
        outcome = status['status'] if status else 'not_run'
        reason = status.get('reason', '') if status else 'Campaign finalized before this case'
        if case:
            for filename, label in [('manual-interruption.json', 'user_interrupted'),
                                    ('warmup-skip.json', 'budget_skipped'),
                                    ('engine-run-skip.json', 'budget_skipped'),
                                    ('finalization-interruption.json', 'finalization_interrupted')]:
                if (case/filename).exists():
                    decision = read(case/filename)
                    outcome = label
                    reason = decision.get('reason', '')
            combined = {}
            for filename in ['engine-observations.jsonl', 'observations.jsonl']:
                for row in rows(case/filename):
                    if row['engine'] in ENGINES:
                        combined[(row['engine'], row['repetition'])] = row
            observations = list(combined.values())
            evidence.extend(dict(row, campaign_index=index, case_directory=str(case.relative_to(ROOT))) for row in observations)
            if outcome == 'error':
                errors = [r.get('error', '') for r in observations if r['status']=='error']
                reason = '; '.join(dict.fromkeys(errors))
                if errors and all('Memory limit exceeded' in err for err in errors):
                    outcome = 'memory_failure'
        else:
            observations = []
        outcomes.append(dict(index=index, **job, outcome=outcome, reason=reason))
        cell = dict(index=index, rule=job['rule'], **job['spec'], outcome=outcome)
        for engine in ENGINES:
            measured = [r for r in observations if r['engine']==engine and not r['warmup']]
            valid = (len(measured)==3 and {r['repetition'] for r in measured}=={1,2,3}
                     and all(r['status'] not in ('failed','error') for r in measured))
            if outcome == 'complete':
                assert valid, (index, engine, 'Missing/invalid repetitions')
            cell[engine+'_complete'] = valid and outcome == 'complete'
            cell[engine+'_active'] = bool(valid and all(r['changed'] for r in measured))
            for scope in ('rewrite','total'):
                values = [r[scope+'_s'] for r in measured] if valid else []
                # Published paired timings require a fully completed case.
                cell[engine+'_'+scope] = stats.median(values) if cell[engine+'_complete'] else None
                cell[engine+'_'+scope+'_min'] = min(values) if cell[engine+'_complete'] else None
                cell[engine+'_'+scope+'_max'] = max(values) if cell[engine+'_complete'] else None
        for engine in ENGINES[1:]:
            for scope in ('rewrite','total'):
                a, b = cell['pyzx_'+scope], cell[engine+'_'+scope]
                cell[engine+'_'+scope+'_speedup'] = a/b if a is not None and b and cell['pyzx_active'] and cell[engine+'_active'] else None
        cells.append(cell)
    counts = Counter(c['outcome'] for c in outcomes)
    save('outcomes.json', outcomes)
    save('timings.json', cells)
    with (OUT/'timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted({k for c in cells for k in c}))
        writer.writeheader(); writer.writerows(cells)
    (OUT/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in evidence))
    for name in ['plan.json','progress.json','finalization.json','skip-decisions.json',
                 'warmup-policy.json','engine-run-policy-1800s.json','engine-run-policy.json']:
        shutil.copy2(ROOT/name, OUT/name)
    # Verify the measured snapshot independently of later plotting/cleanup edits.
    with zipfile.ZipFile(ROOT/'correctness/sources-attempt-0.zip') as archive:
        source_integrity = {
            p: hashlib.sha256(archive.read(p.replace('\\', '/'))).hexdigest() == h
            for p, h in plan['source_hashes'].items()
        }
    working_integrity = {
        p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() == h if (REPO/p).exists() else None
        for p, h in plan['source_hashes'].items()
    }
    case_integrity = {}
    for p in sorted((ROOT/'cases').glob('*/environment.json')):
        env=read(p)
        case_integrity[p.parent.name] = all(plan['source_hashes'].get(name)==h for name,h in env['sources'].items())
    assert all(source_integrity.values()) and all(case_integrity.values()), 'Source drift'
    fixture_rows=[r for r in rows(ROOT/'correctness/observations.jsonl') if r['stage']=='correctness']
    fixture_counts=Counter(r['status'] for r in fixture_rows)
    good=[c for c in cells if c['outcome']=='complete']
    circuits=[c for c in good if c['rule']=='full_reduce']
    rules=[c for c in good if c['rule']!='full_reduce']
    families=sorted({c['family'] for c in circuits})
    latest=[max([c for c in rules if c['family']==family],key=lambda c:c['size']) for family in sorted({c['family'] for c in rules})]
    families128=[]
    for family in families:
        selected=[c for c in circuits if c['family']==family and c['size']==128]
        record=dict(family=family)
        for engine in ENGINES[1:]:
            for scope in ('rewrite','total'):
                record[engine+'_'+scope]=stats.geometric_mean(c[engine+'_'+scope+'_speedup'] for c in selected)
        families128.append(record)
    wins128={e:{scope:sum(c[e+'_'+scope+'_speedup']>1 for c in circuits if c['size']==128) for scope in ['rewrite','total']} for e in ENGINES[1:]}
    rulewins={e:{scope:sum(c[e+'_'+scope+'_speedup'] is not None and c[e+'_'+scope+'_speedup']>1 for c in latest) for scope in ['rewrite','total']} for e in ENGINES[1:]}
    measured=[r for r in evidence if not r['warmup'] and outcomes[r['campaign_index']]['outcome']=='complete']
    checks={e:dict(Counter(r.get('check',{}).get('status',r['status']) for r in measured if r['engine']==e)) for e in ENGINES}
    summary=dict(finalized_at=state['finalized_at'],configurations=dict(counts),complete_circuits=len(circuits),
        complete_rules=len(rules),complete_measured_engine_trials=len(measured),correctness=dict(fixture_counts),
        measured_semantics=checks,circuit_128_wins=wins128,largest_rule_wins=rulewins,largest_rules=latest,
        circuit_128_geomean_speedups=families128,
        source_hashes_unchanged=all(v is True for v in working_integrity.values()))
    save('summary.json',summary)
    save('integrity.json',dict(working_sources=working_integrity,archived_sources=source_integrity,case_sources=case_integrity,
         all_complete_cases_have_exactly_three_successful_measurements_per_engine=True))
    text=['# Final benchmark results — 29 October 2026','',
      f"Campaign stopped at the user's request at {state['finalized_at']}. No benchmark or cutoff monitor remains active. The dedicated database was cleaned of the interrupted graph and stopped.",'',
      '**Main finding:** database batching produces substantial full-reduce speedups at 128 qubits and strong wins for selected large individual rules. It does not produce universal standalone-rule superiority. Import/export overhead often removes rewrite-only wins.','',
      '## Coverage','']
    text+=table(['Outcome','Configurations'],sorted(counts.items()))
    text += [f'{len(circuits)} completed circuit configurations and {len(rules)} completed standalone-rule configurations; {len(measured):,} measured engine trials, plus one warmup per engine per completed configuration. All 90 circuit configurations at 3–128 qubits completed. No 256-qubit circuit configuration completed.','',
      'The full planned grid contained 257 configurations. All 28 workload shapes were exercised, but the largest planned sizes were not all reached. Cases marked not_run were never started; interrupted and skipped cases retain partial observations but contribute no paired timing medians.','',
      '## Full reduce at 128 qubits','',
      'Speedup = PyZX time / database time; values above one favor the database. Entries are geometric means of three per-seed median ratios. Each median uses three measured repetitions. Total time includes import/rewrite/export; PyZX total includes graph copying.','']
    text+=table(['Family','Cypher rewrite','Hybrid rewrite','Cypher total','Hybrid total'],
      [[c['family'],*[fmt(c[k])+'×' for k in ['cypher_rewrite','hybrid_rewrite','cypher_total','hybrid_total']]] for c in families128])
    text += [f"At 128 qubits Cypher wins {wins128['cypher']['rewrite']}/15 rewrite and {wins128['cypher']['total']}/15 total-time configurations. Hybrid wins {wins128['hybrid']['rewrite']}/15 rewrite and {wins128['hybrid']['total']}/15 total-time configurations. The two hybrid total-time losses are local-layered inputs.",'',
      '![Circuit total-time scaling](circuit-total-speedups.png)','',
      '## Largest completed size of each standalone workload','',
      'These are the largest **completed**, not largest planned, sizes. Sizes mean motif copies for copies_* workloads, chain length for identity_chain, and the generator size parameter otherwise. Different families therefore have different graph sizes. These selected largest sizes are not a common-size comparison.','']
    text+=table(['Workload','Size','Cypher rewrite','Hybrid rewrite','Cypher total','Hybrid total'],
      [[c['family'],c['size'],*[fmt(c[k])+'×' if c[k] is not None else '—' for k in ['cypher_rewrite_speedup','hybrid_rewrite_speedup','cypher_total_speedup','hybrid_total_speedup']]] for c in latest])
    text += [f"Across these 23 workload shapes, Cypher wins {rulewins['cypher']['rewrite']}/23 on rewrite and {rulewins['cypher']['total']}/23 on total time; hybrid wins {rulewins['hybrid']['rewrite']}/23 and {rulewins['hybrid']['total']}/23 respectively.",'',
      '- Identity copies at 8,192 show about 411–412× rewrite and 23–24× total-time speedups. Spider-fusion copies at 4,096 show about 119–125× rewrite and 11× total-time speedups.',
      '- Gadget-pivot fans at size 1,024 show about 72–76× rewrite and 3.5× total speedups. This is a particularly clear case where the larger workload reveals an end-to-end database advantage.',
      '- For 8,192 local-complementation copies, hybrid wins about 190× on rewrite and 10.4× on total time; pure Cypher loses. The native implementation materially changes this result.',
      '- For 8,192 gadget-fusion copies, both database variants win about 9× on rewrite but lose on total time. Graph transport dominates the benefit.',
      '- Disjoint bialgebra, copy, Hadamard cancellation and supplementarity remain poor database workloads in this campaign. Larger inputs did not rescue them. Dense bialgebra at sizes 512 and 1,024 exceeded available memory.','',
      '![Standalone total-time speedups](rule-total-speedups.png)','',
      '![Selected scaling curves](selected-scaling.png)','',
      '## Correctness, integrity, and interpretation','',
      '- All 88 selected live regression tests passed. Of 446 fixture checks (223 inputs × two database modes), 440 verified and six remained inconclusive; none failed. No supplemental proof campaign was performed for those six in this run.',
      f'- Completed measured outputs have input-semantic check counts: `{json.dumps(checks, sort_keys=True)}`. Inconclusive does not mean correct. Structural differences and degree-sequence agreement are not semantic certificates.',
      '- The timed engine code, workload generators, and verification sources match the original campaign hashes across every case. The finalizer checks exact repetition completeness and publishes its integrity results.',
      '- Limits changed during the campaign: initially two hours per engine; then 30 minutes for warmups, then for all engine runs; finally five minutes for all engine runs. The external cutoff includes verification before the completion event and polls every five seconds. Therefore the result set is censored by changing budgets, not an unbiased comparison at one fixed limit.',
      '- Long measurements completed before a policy change are retained. For example, copies_bialgebra at 4,096 completed a Cypher repetition in about 55 minutes while its median was about 13.5 minutes. Per-repetition min/max timings in the CSV expose this variability; the cause was not established.',
      '- Local-layered seed labels generate identical inputs; do not treat them as three independent circuit instances. Deterministic synthetic rule workloads also do not establish population-level confidence intervals.',
      '- No claim of asymptotic superiority or fully verified semantics for the large workloads follows from these timings. Curves show observed finite-size scaling; favorable cases should be presented together with losses and memory/cutoff outcomes.','',
      '## Incomplete and skipped cases','']
    text+=table(['Index','Workload','Size','Outcome'],[[c['index'],c['spec']['family'],c['spec']['size'],c['outcome']] for c in outcomes if c['outcome']!='complete'])
    text += ['## Files','',
      '- `timings.csv` / `timings.json`: complete-case medians, min/max repetition times, active/no-op flags and paired ratios for every planned configuration.',
      '- `observations.jsonl`: frozen union of engine-level and enriched comparison observations, including partial cases. Case index and source directory are included.',
      '- `outcomes.json`: explicit classification and failure reasons for every configuration.',
      '- `summary.json`, `integrity.json`, `evidence-manifest.json`: machine-readable findings and hashes of source evidence.',
      '- PNG and SVG figures: circuit total-speedup curves, largest completed rule speeds, and selected total-runtime scaling. These show medians/geometric means, not confidence intervals.',
      '- Original raw logs, graph snapshots and source archives remain in `../cases/`; correctness evidence remains in `../correctness/`. Earlier campaigns were not pooled.','']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    colors={'pyzx':'#616b75','cypher':'#1769aa','hybrid':'#d67414'}
    fig,axes=plt.subplots(1,5,figsize=(16,4),sharey=True,layout='constrained')
    for ax,family in zip(axes,families):
        for engine in ENGINES[1:]:
            sizes=sorted({c['size'] for c in circuits if c['family']==family})
            ys=[stats.geometric_mean(c[engine+'_total_speedup'] for c in circuits if c['family']==family and c['size']==n) for n in sizes]
            ax.plot(sizes,ys,'o-',label=engine.capitalize(),color=colors[engine])
        ax.axhline(1,color='#999',ls='--',lw=1); ax.set(xscale='log',yscale='log',title=family.replace('random_','').replace('layered_','Layered\n'),xlabel='Qubits')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Total-time speedup (PyZX / database)'); axes[-1].legend()
    for suffix in ['png','svg']:fig.savefig(OUT/f'circuit-total-speedups.{suffix}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,10),layout='constrained')
    for engine,offset in [('cypher',-.15),('hybrid',.15)]:
        ax.scatter([c[engine+'_total_speedup'] for c in latest],[i+offset for i in range(len(latest))],color=colors[engine],label=engine.capitalize(),s=35)
    ax.axvline(1,color='#777',ls='--'); ax.set(xscale='log',yticks=range(len(latest)),yticklabels=[f"{c['family']} ({c['size']:,})" for c in latest],xlabel='Total-time speedup (PyZX / database)',title='Largest completed scale per standalone workload')
    ax.invert_yaxis(); ax.grid(axis='x',alpha=.2); ax.legend()
    for suffix in ['png','svg']:fig.savefig(OUT/f'rule-total-speedups.{suffix}')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,family in zip(axes.flat,['copies_identity','copies_spider_fusion','pivot_gadget','copies_lcomp']):
        selected=sorted([c for c in rules if c['family']==family],key=lambda c:c['size'])
        for engine in ENGINES:
            ax.plot([c['size'] for c in selected],[c[engine+'_total'] for c in selected],'o-',color=colors[engine],label=engine.capitalize())
        ax.set(xscale='log',yscale='log',title=family,xlabel='Workload size parameter',ylabel='Median total time (seconds)'); ax.grid(alpha=.2)
    axes.flat[0].legend()
    for suffix in ['png','svg']:fig.savefig(OUT/f'selected-scaling.{suffix}')
    plt.close(fig)
    evidence_paths=sorted((ROOT/'cases').glob('*/engine-observations.jsonl'))+sorted((ROOT/'cases').glob('*/observations.jsonl'))+[ROOT/'correctness/observations.jsonl']
    save('evidence-manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_paths})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('largest_rules',)},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT,
                        help='Fresh destination; existing frozen results are never overwritten')
    OUT = parser.parse_args().output.resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    main()
