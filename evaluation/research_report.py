"""Hierarchical paired summaries and native PGFPlots (no tikzplotlib needed)."""
import csv
import json
import math
from pathlib import Path
import random
import statistics
import hashlib
from collections import defaultdict


def interval(clusters, seed=17, samples=2000):
    """Geometric mean of per-input median paired speedups; hierarchical CI."""
    rng = random.Random(seed)
    logs = [[math.log(x) for x in cluster] for cluster in clusters]
    point = math.exp(statistics.mean(statistics.median(c) for c in logs))
    draws = []
    for _ in range(samples):
        selected = [rng.choice(logs) for _ in logs]
        draws.append(math.exp(statistics.mean(statistics.median(
            rng.choices(c, k=len(c))) for c in selected)))
    draws.sort()
    return point, draws[int(.025*samples)], draws[min(samples-1, int(.975*samples))]


def summarize(plan, rows):
    groups = defaultdict(list)
    for spec in plan['cases']:
        groups[(spec['family'], spec['size'], spec['depth'], spec['scale'])].append(spec)
    trials = defaultdict(list)
    warmup_failures = defaultdict(int)
    attempt_failures = defaultdict(int)
    inputs = {}
    stops = defaultdict(list)
    for row in rows:
        cid = row['case']['id']
        if row['kind'] == 'trial' and not row['warmup']:
            trials[cid].append(row)
        elif row['kind'] == 'trial' and row['warmup'] and row['status'] != 'ok':
            warmup_failures[cid] += 1
        elif row['kind'] == 'input':
            inputs[cid] = row
        elif row['kind'] in ('limit_stop', 'adaptive_skip', 'case_budget'):
            stops[cid].append(row)
        elif row['kind'] in ('error', 'timeout'):
            attempt_failures[cid] += 1
        elif row['kind'] == 'cleanup_error':
            # Cleanup following a primary failure is not a second attempt.
            attempt_failures[cid] = max(1, attempt_failures[cid])
    result = []
    for (family, size, depth, scale), specs in sorted(groups.items()):
        selected = [r for spec in specs for r in trials[spec['id']]]
        good = [r for r in selected if r['status'] == 'ok']
        failed = sum(r['status'] != 'ok' for r in selected)
        verified = sum(all(e['verification']['status'] == 'verified' for e in r['engines'].values()) for r in good)
        expected = len(specs)*plan['repetitions']
        bad_warmups = sum(warmup_failures[s['id']] for s in specs)
        failed_attempts = sum(attempt_failures[s['id']] for s in specs)
        versions = {r.get('source_version', 'legacy') for r in selected}
        mixed_versions = len(versions) > 1
        complete = len(selected) == expected and not failed and not bad_warmups and not failed_attempts and not mixed_versions
        by_input = defaultdict(list)
        for spec in specs:
            # Some seeds produce identical inputs (especially two-qubit locality
            # controls). Do not count these as independent sampled circuits.
            by_input[inputs.get(spec['id'], {}).get('sha256', spec['id'])].extend(trials[spec['id']])
        item = dict(family=family, size=size, depth=depth, scale=scale,
                    expected_trials=expected, recorded_trials=len(selected), failed_trials=failed,
                    failed_warmups=bad_warmups,
                    failed_attempts=failed_attempts, mixed_source_versions=mixed_versions,
                    verified_trials=verified, inconclusive_trials=len(good)-verified,
                    planned_seeds=len(specs),
                    generated_instances=sum(s['id'] in inputs for s in specs),
                    measured_instances=sum(bool(trials[s['id']]) for s in specs),
                    distinct_inputs=len({inputs[s['id']]['sha256'] for s in specs if s['id'] in inputs}),
                    complete=complete,
                    vertices=statistics.median(inputs[s['id']]['metrics']['vertices'] for s in specs
                                               if s['id'] in inputs) if any(s['id'] in inputs for s in specs) else None,
                    db_s=None, pyzx_s=None, speedup=None, ci_low=None, ci_high=None)
        for kind in ('limit_stop', 'adaptive_skip', 'case_budget'):
            item[kind+'_cases'] = sum(any(r['kind'] == kind for r in stops[s['id']]) for s in specs)
        # Capacity evidence includes successful engine results whose partner
        # exceeded a budget. Never turn censoring into a fabricated speedup.
        for engine in ('db', 'pyzx'):
            completed = [r['engines'][engine]['reduce_s'] for r in selected
                         if r['engines'][engine].get('reduce_s') is not None
                         and r['engines'][engine]['verification']['status'] != 'failed'
                         and r['status'] in ('ok', 'resource_limit')]
            item[engine+'_completed_trials'] = len(completed)
            item[engine+'_completed_median_s'] = statistics.median(completed) if completed and not mixed_versions else None
        for engine in ('db', 'pyzx'):
            for status in ('timeout', 'memory_limit', 'not_run_after_timeout'):
                item[engine+'_'+status+'_trials'] = sum(r['engines'][engine].get('status') == status for r in selected)
        for level, statuses in [('tensor', ('verified', 'failed', 'inconclusive')),
                                ('graph', ('equal', 'different', 'inconclusive')),
                                ('degree_sequence', ('equal', 'different'))]:
            for status in (*statuses, 'not_recorded'):
                item[f'outputs_{level}_{status}'] = sum(
                    r.get('comparison', {}).get(level, {}).get('status', 'not_recorded') == status
                    for r in selected)
        if complete:
            clusters = [[r['engines']['pyzx']['reduce_s']/r['engines']['db']['reduce_s']
                         for r in cluster] for cluster in by_input.values()]
            item['speedup'], item['ci_low'], item['ci_high'] = interval(clusters)
            for engine in ('db', 'pyzx'):
                item[engine+'_s'] = statistics.median(statistics.median(
                    r['engines'][engine]['reduce_s'] for r in cluster) for cluster in by_input.values())
        item['evidence'] = 'failed_or_incomplete' if not complete else 'verified' if verified == expected else 'inconclusive'
        result.append(item)
    return result


def report(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    rows = []
    env_path = directory / 'environment.json'
    environment = json.loads(env_path.read_text()) if env_path.exists() else {}
    legacy_version = hashlib.sha256(json.dumps(dict(source=environment.get('source_hashes', {}),
                                                   packages=environment.get('packages', {})), sort_keys=True).encode()).hexdigest() if environment else 'legacy'
    for path in sorted((directory / 'blocks').glob('*.jsonl')):
        content = path.read_text(encoding='utf-8')
        lines = content.splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
                row['source_version'] = row.get('source_version') or legacy_version
                rows.append(row)
            except json.JSONDecodeError:
                # A killed worker can leave a partial final line; no trial is invented.
                if index != len(lines)-1 or content.endswith('\n'):
                    raise ValueError(f'Corrupt completed record in {path}:{index+1}')
    summary = summarize(plan, rows)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if summary:
        with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    figures = directory / 'figures'; figures.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in summary:
        if plan.get('profile') == 'focused' and row['family'].startswith(('random_', 'layered_')):
            groups[(row['family'], f"qubits{row['size']}")].append(dict(row, size=row['depth'], scale='depth'))
        else:
            groups[(row['family'], row['depth'])].append(row)
    panels = []
    for (family, depth), values in sorted(groups.items()):
        name = family + (('-'+depth if isinstance(depth, str) else f'-depth{depth}') if depth is not None else '')
        values.sort(key=lambda r: r['size'])
        # Missing/failed cells stay nan, breaking curves rather than joining across them.
        lines = ['size vertices db pyzx speedup lower upper db_completed pyzx_completed']
        for row in values:
            numbers = [row[k] for k in ('size', 'vertices', 'db_s', 'pyzx_s', 'speedup', 'ci_low', 'ci_high',
                                       'db_completed_median_s', 'pyzx_completed_median_s')]
            lines.append(' '.join('nan' if v is None else str(v) for v in numbers))
        (figures / f'{name}.dat').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        title = name.replace('_', r'\_')
        axis = (r'\begin{tikzpicture}'+'\n'+r'\begin{axis}[width=7cm,height=5cm,xmode=log,ymode=log,'
                r'log basis x=2,unbounded coords=jump,xlabel={' + values[0]['scale'].replace('_', r'\_') +
                r'},ylabel={Reduction time (s)},title={' + title + r'},legend pos=north west]'+'\n'+
                r'\addplot+[mark=square*] table[x=size,y=db] {'+name+r'.dat};\addlegendentry{ZXdb}'+'\n'+
                r'\addplot+[mark=*] table[x=size,y=pyzx] {'+name+r'.dat};\addlegendentry{PyZX}'+'\n'+
                r'\end{axis}'+'\n'+r'\end{tikzpicture}'+'\n')
        (figures / f'{name}.tikz').write_text(axis, encoding='utf-8')
        capacity = axis.replace('y=db]', 'y=db_completed]').replace('y=pyzx]', 'y=pyzx_completed]')
        capacity = capacity.replace('Reduction time (s)', 'Completed attempts only (s)')
        (figures / f'{name}-capacity.tikz').write_text(capacity, encoding='utf-8')
        speed = (r'\begin{tikzpicture}\begin{axis}[width=7cm,height=5cm,xmode=log,ymode=log,'
                 r'log basis x=2,unbounded coords=jump,xlabel={' + values[0]['scale'].replace('_', r'\_') +
                 r'},ylabel={PyZX / ZXdb},title={' + title + r'}]'+'\n'+
                 r'\addplot+[mark=*,error bars/.cd,y dir=both,y explicit] table[x=size,y=speedup,'
                 r'y error minus expr=\thisrow{speedup}-\thisrow{lower},'
                 r'y error plus expr=\thisrow{upper}-\thisrow{speedup}] {'+name+r'.dat};'+'\n'+
                 r'\addplot[black,dashed,domain='+str(values[0]['size'])+':'+str(max(values[-1]['size'], values[0]['size']+1))+
                 r'] {1};\end{axis}\end{tikzpicture}'+'\n')
        (figures / f'{name}-speedup.tikz').write_text(speed, encoding='utf-8')
        panels.append(r'\input{'+name+r'.tikz}\quad'+'\n'+r'\input{'+name+r'-speedup.tikz}\par\medskip')
        panels.append(r'\noindent '+title+': '+str(sum(r['recorded_trials'] for r in values))+'/'+
                      str(sum(r['expected_trials'] for r in values))+' measured pairs; '+
                      str(sum(r['failed_trials'] for r in values))+' non-ok measured pairs; '+
                      str(sum(r['failed_attempts'] for r in values))+' failed attempts; '+
                      str(sum(r['adaptive_skip_cases'] for r in values))+' adaptive skips.'+r'\par')
        if any(not r['complete'] for r in values):
            panels.append(r'\noindent Incomplete cells have no paired speedup. Available engine timings follow (completed attempts only).\par'+
                          r'\input{'+name+r'-capacity.tikz}\par\medskip')

    # Compilable article; fragments can also be copied into a paper's figure environment.
    document = (r'\documentclass{article}'+'\n'+r'\usepackage[margin=1cm]{geometry}'+'\n'+
                r'\usepackage{pgfplots}\pgfplotsset{compat=1.18}'+'\n'+r'\begin{document}'+'\n'+
                r'\noindent Timing curves include semantically inconclusive large cases. '
                r'Check summary.csv before making correctness or speedup claims. '
                r'Bars are hierarchical bootstrap 95\% intervals; one-input bars describe timing noise only.'+'\n\n'+
                '\n'.join(panels)+'\n'+r'\end{document}'+'\n')
    (figures / 'figures.tex').write_text(document, encoding='utf-8')
    # Eight rule panels in the paper's order, using the controlled fan/pattern
    # families. This is a standalone 4-by-2 figure, with a common legend.
    paper_order = ['pivot', 'pivot_gadget', 'lcomp', 'gadget_fusion',
                   'spider_fusion', 'identity', 'bialgebra', 'pivot_boundary']
    panels8 = []
    for family in paper_order:
        if (family, None) not in groups and ('copies_'+family, None) in groups:
            family = 'copies_'+family
        values = groups.get((family, None))
        if not values:
            panels8.append(r'\nextgroupplot[group/empty plot]')
            continue
        label = family.replace('_', r'\_')
        panels8.append(r'\nextgroupplot[title={'+label+r'},xlabel={'+values[0]['scale'].replace('_', r'\_')+r'}]'+'\n'+
                       r'\addplot+[mark=square*] table[x=size,y=db] {'+family+r'.dat};'+'\n'+
                       r'\addplot+[mark=*] table[x=size,y=pyzx] {'+family+r'.dat};')
    combined = (r'\documentclass[tikz,border=3pt]{standalone}'+'\n'+
                r'\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}'+'\n'+
                r'\begin{document}\begin{tikzpicture}'+'\n'+
                r'\begin{groupplot}[group style={group size=4 by 2,horizontal sep=1.4cm,vertical sep=1.8cm},'
                r'width=5cm,height=4.5cm,xmode=log,ymode=log,log basis x=2,unbounded coords=jump,'
                r'ylabel={Time (s)}]'+'\n'+'\n'.join(panels8)+'\n'+
                r'\end{groupplot}\end{tikzpicture}\end{document}'+'\n')
    (figures / 'paper_rules.tex').write_text(combined, encoding='utf-8')
    (directory / 'REPORT.md').write_text(
        '# Campaign results\n\n' +
        f'{len(summary)} cells; {sum(r["complete"] for r in summary)} complete; '
        f'{sum(r["evidence"] == "verified" for r in summary)} fully tensor-verified.\n\n'
        'Inspect summary.csv for failures, missing trials and inconclusive validation. '
        'Failed/incomplete cells have no aggregated speedup. Raw observations are in blocks/*.jsonl.\n\n'
        'Adaptive limit_stop_cases count both-engine timeout stops; adaptive_skip_cases are unrun cells. '
        'case_budget_cases are operational limits, not evidence that both simplifiers timed out.\n\n'
        'Compile from figures/: `pdflatex figures.tex`. Each .tikz fragment has a matching .dat file. '
        'Keep both together when including in a paper. These figures are current-implementation '
        'controlled studies, not an exact reproduction of historical Figure 16.\n', encoding='utf-8')
    from evaluation.research_scale import write_scale_artifacts
    write_scale_artifacts(directory, plan, summary, rows)
    print(f'Report and PGFPlots sources: {directory}')
