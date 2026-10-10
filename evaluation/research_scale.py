"""Full-grid and explicitly conditional completed-pair scale summaries."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics


def stratified_interval(strata, samples):
    point = math.exp(statistics.mean(
        statistics.mean(statistics.median(c) for c in stratum) for stratum in strata))
    rng = random.Random(20260910)
    draws = []
    for _ in range(samples):
        means = []
        for stratum in strata:
            selected = rng.choices(stratum, k=len(stratum))
            means.append(statistics.mean(statistics.median(rng.choices(c, k=len(c))) for c in selected))
        draws.append(math.exp(statistics.mean(means)))
    draws.sort()
    return point, draws[int(.025*samples)], draws[min(samples-1, int(.975*samples))]


def usable_pair(row):
    if row.get('status') != 'ok':
        return False
    for engine in ('db', 'pyzx'):
        value = row['engines'][engine]
        seconds = value.get('reduce_s')
        if value.get('verification', {}).get('status') == 'failed' or seconds is None or not math.isfinite(seconds) or seconds <= 0:
            return False
    return True


def large_scale_summary(plan, summary, rows, samples=2000):
    policy = plan.get('large_scale_summary')
    if not policy:
        return []
    inputs = {r['case']['id']: r['sha256'] for r in rows if r['kind'] == 'input'}
    trials = defaultdict(list)
    for row in rows:
        if row['kind'] == 'trial' and not row['warmup']:
            trials[row['case']['id']].append(row)
    result = []
    for family in policy['families']:
        sizes = policy['sizes']
        cells = [r for r in summary if r['family'] == family and r['size'] in sizes and r['depth'] is None]
        specs = [s for s in plan['cases'] if s['family'] == family and s['size'] in sizes]
        complete = len(cells) == len(sizes) and all(r['complete'] for r in cells)
        item = dict(family=family, sizes=sizes, complete=complete,
                    planned_instances=len(specs),
                    measured_instances=sum(bool(trials[s['id']]) for s in specs),
                    distinct_inputs=len({inputs[s['id']] for s in specs if s['id'] in inputs}),
                    measured_pairs=sum(len(trials[s['id']]) for s in specs),
                    speedup=None, ci_low=None, ci_high=None,
                    reason=None if complete else 'Missing, failed or resource-limited size cells; no survivor-only average')
        if complete:
            strata = []
            for size in sizes:
                grouped = defaultdict(list)
                for spec in specs:
                    if spec['size'] != size:
                        continue
                    for row in trials[spec['id']]:
                        grouped[inputs.get(spec['id'], spec['id'])].append(math.log(
                            row['engines']['pyzx']['reduce_s']/row['engines']['db']['reduce_s']))
                strata.append(list(grouped.values()))
            item['speedup'], item['ci_low'], item['ci_high'] = stratified_interval(strata, samples)
        # Secondary estimator: every successful measured pair, including a partial
        # input checkpoint. Keep it separate from the full-grid estimand above.
        grouped = defaultdict(lambda: defaultdict(list))
        used_ids, versions, pairs = set(), set(), 0
        for spec in specs:
            for row in trials[spec['id']]:
                if not usable_pair(row):
                    continue
                fingerprint = inputs.get(spec['id'], spec['id'])
                grouped[spec['size']][fingerprint].append(math.log(
                    row['engines']['pyzx']['reduce_s']/row['engines']['db']['reduce_s']))
                used_ids.add(fingerprint)
                versions.add(row.get('source_version', 'legacy'))
                pairs += 1
        item.update(completed_sizes=sorted(grouped), completed_distinct_inputs=len(used_ids),
                    completed_pairs=pairs, completed_speedup=None, completed_ci_low=None,
                    completed_ci_high=None, completed_mixed_versions=len(versions) > 1,
                    completed_basis='Successful measured pairs only; conditional on completion')
        if grouped and len(versions) <= 1:
            strata = [list(grouped[size].values()) for size in sorted(grouped)]
            item['completed_speedup'], item['completed_ci_low'], item['completed_ci_high'] = stratified_interval(strata, samples)
        result.append(item)
    return result


def write_scale_artifacts(directory, plan, summary, rows):
    if not plan.get('large_scale_summary'):
        return
    directory = Path(directory)
    result = large_scale_summary(plan, summary, rows)
    (directory/'large_scale_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (directory/'large_scale_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0])); writer.writeheader(); writer.writerows(result)
    labels = dict(copies_identity='Identity', copies_lcomp='Local complementation',
                  copies_spider_fusion='Spider fusion', copies_pivot='Pivot',
                  copies_gadget_fusion='Gadget fusion', copies_bialgebra='Bialgebra',
                  copies_pivot_boundary='Boundary pivot', copies_pivot_gadget='Gadget pivot',
                  random_clifford_t='Full reduction')
    # Report-only exploratory selections never modify the saved benchmark plan.
    options_path = directory/'report-options.json'
    options = json.loads(options_path.read_text()) if options_path.exists() else {}
    exploratory = options.get('exploratory_sizes', {})
    display = list(result)
    for family, selected_sizes in exploratory.items():
        if family not in plan['large_scale_summary']['families'] or not selected_sizes or not set(selected_sizes) <= set(plan['large_scale_summary']['sizes']):
            raise ValueError('Exploratory sizes must be a nonempty subset of the declared sizes for a declared family')
        selected_plan = dict(plan, large_scale_summary=dict(families=[family], sizes=selected_sizes))
        replacement = large_scale_summary(selected_plan, summary, rows)[0]
        replacement['selection'] = 'Exploratory: size range chosen after inspecting results'
        display = [replacement if item['family'] == family else item for item in display]
    all_sizes = options.get('table_scope') == 'all_planned_sizes'
    if all_sizes:
        from evaluation.research_report import summarize
        if exploratory:
            raise ValueError('All-size tables cannot also select exploratory subsets')
        display = []
        for family in labels:
            specs = [s for s in plan['cases'] if s['family'] == family]
            if not specs:
                continue
            circuit = family.startswith(('random_', 'layered_'))
            selected_rows = [r for r in rows if r['case']['family'] == family]
            if circuit:
                # Reuse the size-stratified estimator on circuit depth in memory.
                specs = [dict(s, size=s['depth'], depth=None) for s in specs]
                selected_rows = [dict(r, case=dict(r['case'], size=r['case']['depth'], depth=None)) for r in selected_rows]
            selected_plan = dict(plan, cases=specs, large_scale_summary=dict(
                families=[family], sizes=sorted({s['size'] for s in specs})))
            estimate = large_scale_summary(selected_plan, summarize(selected_plan, selected_rows), selected_rows)[0]
            estimate['size_axis'] = 'circuit generation depth (8 qubits)' if circuit else 'copies'
            display.append(estimate)
    (directory/'large_scale_display_summary.json').write_text(json.dumps(display, indent=2), encoding='utf-8')

    def table_text(table_items, exploratory):
        note = r'$^{\dagger}$'
        lines = [r'\begin{tabular}{llrrrr}', r'\hline',
                 r'Rule & Sizes used & Inputs & Pairs & Speedup & 95\% interval \\', r'\hline']
        for item in table_items:
            available = item['completed_speedup'] is not None
            precision = 3 if available and item['completed_speedup'] < 0.1 else 2
            speed = f"{item['completed_speedup']:.{precision}f}" if available else '--'
            interval = f"[{item['completed_ci_low']:.{precision}f}, {item['completed_ci_high']:.{precision}f}]" if available else (
                'Mixed versions' if item['completed_mixed_versions'] else 'No paired data')
            sizes_used = ', '.join(map(str, item['completed_sizes'])) or '--'
            coverage = f"{item['completed_distinct_inputs']}/{item['planned_instances']}"
            lines.append(f"{labels[item['family']]}{note if item['family'] in exploratory else ''} & {sizes_used} & {coverage} & {item['completed_pairs']} & {speed} & {interval}" + r' \\')
        lines += [r'\hline', r'\end{tabular}',
                  r'\par\smallskip\noindent Completed measured pairs only. Inputs shows used distinct inputs / planned cases. '
                  r'Missing and timed-out pairs are excluded. Intervals are conditional on completion; '
                  r'they do not account for selection bias from missing results.']
        if any(item.get('size_axis') for item in table_items):
            lines.append(r'\par\noindent All planned sizes are eligible, starting at 32. Rule sizes are copy counts; full-reduction sizes are generation depths on eight qubits. Only represented sizes enter each average, with equal size weight. Missing larger sizes are not measured performance limits.')
        if exploratory:
            lines.append(r'\par\noindent $^{\dagger}$Exploratory size selection made after inspecting results; '
                         r'the excluded smaller sizes remain in the curves and the predeclared table. '
                         r'An interval including one does not establish a speed advantage.')
        return '\n'.join(lines)+'\n'

    (directory/'large_scale_table_predeclared.tex').write_text(table_text(result, {}), encoding='utf-8')
    (directory/'large_scale_table.tex').write_text(table_text(display, exploratory), encoding='utf-8')
    if all_sizes:
        (directory/'all_cases_table.tex').write_text(table_text(display, {}), encoding='utf-8')
        (directory/'all_cases_summary.json').write_text(json.dumps(display, indent=2), encoding='utf-8')
        main = directory/'figures'/'figures.tex'
        main.write_text(main.read_text().replace(r'\end{document}', r'\clearpage\input{../all_cases_table.tex}'+'\n'+r'\end{document}'), encoding='utf-8')
    sizes = ', '.join(map(str, plan['large_scale_summary']['sizes']))
    document = [r'\documentclass{article}', r'\usepackage[margin=1cm]{geometry}',
                r'\usepackage{pgfplots}\pgfplotsset{compat=1.18}', r'\begin{document}',
                r'\noindent Independent-copy workloads; reduction time excludes loading and reading. '
                r'Large graphs may have inconclusive semantic validation. Missing pairs are not assigned speedups.']
    for family in plan['large_scale_summary']['families']:
        if any(r['family'] == family for r in summary):
            document += [r'\input{'+family+r'.tikz}\quad\input{'+family+r'-speedup.tikz}\par\medskip']
    document += [r'\clearpage\noindent Predeclared large sizes: '+sizes+r' copies. '
                 r'Completed-pair geometric mean speedup gives equal weight to represented sizes, then distinct inputs within size. '
                 r'Intervals resample inputs and their paired timing repetitions within each fixed size. '
                 r'The table is conditional on successful completion. Full-grid estimates remain separate in JSON/CSV and are absent if incomplete.\par\medskip',
                 (r'\input{../large_scale_table_predeclared.tex}' if all_sizes else r'\input{../large_scale_table.tex}'), r'\end{document}']
    (directory/'figures'/'five_rules.tex').write_text('\n'.join(document)+'\n', encoding='utf-8')
