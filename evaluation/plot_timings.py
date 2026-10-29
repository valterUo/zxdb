"""Generate paper-style TikZ/PGFPlots figures from one timings.csv (stdlib only)."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/generous-rerun-20261029/finalized/timings.csv'
ENGINES = {
    'pyzx': ('PyZX', 'black', '*', 'solid'),
    'cypher': ('Cypher', 'blue!70!black', 'square*', 'dashed'),
    'hybrid': ('Cypher+C++', 'orange!85!black', 'triangle*', 'dashdotted'),
}
NAMES = {
    'pivot': 'Pivot', 'pivot_gadget': 'Gadget pivot',
    'lcomp': 'Local complementation', 'gadget_fusion': 'Phase-gadget fusion',
    'spider_fusion': 'Spider fusion', 'identity': 'Identity removal',
    'bialgebra': 'Bialgebra', 'pivot_boundary': 'Boundary pivot',
    'hadamard_cancel': 'Hadamard cancellation', 'copy': 'Copy',
    'supplementarity': 'Supplementarity', 'identity_chain': 'Identity chain',
    'random_clifford': 'Clifford', 'random_clifford_t': 'Clifford+T',
    'random_t_heavy': 'T-heavy', 'layered_local': 'Local layered',
    'layered_global': 'Global layered',
}


def value(row, engine, scope, speedup=False):
    """Never turn failed, missing, nonpositive, or no-op ratios into wins."""
    required = (engine, 'pyzx') if speedup else (engine,)
    for name in required:
        if row[name + '_complete'] != 'True':
            return None
        if speedup and row[name + '_active'] != 'True':
            return None
    try:
        result = float(row[engine + '_' + scope])
        if speedup:
            baseline = float(row['pyzx_' + scope])
            if not math.isfinite(baseline) or baseline <= 0:
                return None
            result = baseline / result
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def summarize(rows, engine, scope, speedup=False):
    values = [value(r, engine, scope, speedup) for r in rows]
    # Require the entire group; do not silently summarize only survivors.
    if any(v is None for v in values):
        return None
    return geometric_mean(values), min(values), max(values)


def select_completed(rows):
    """Keep a common, contiguous completed size range for all three engines."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['family']][int(row['size'])].append(row)
    selected, coverage = [], []
    for family, sizes in sorted(grouped.items()):
        stopped = False
        for size, group in sorted(sizes.items()):
            valid = all(r.get('outcome', 'complete') == 'complete'
                        and all(value(r, e, scope) is not None for e in ENGINES
                                for scope in ('rewrite', 'total')) for r in group)
            included = valid and not stopped
            if included:
                selected.extend(group)
            else:
                stopped = True
            coverage.append(dict(family=family, size=size, included=included,
                                 configurations=len(group),
                                 reason='complete paired timings' if included else
                                 ('incomplete/invalid group' if not valid else 'beyond first incomplete size'),
                                 outcomes=','.join(sorted({r.get('outcome', 'unknown') for r in group}))))
    return selected, coverage


def enrich_and_check(rows, observations_path):
    """Recover input vertices and validate finalized medians against raw trials."""
    if not observations_path.exists():
        if any(not r.get('vertices') for r in rows if r.get('outcome') == 'complete'):
            raise ValueError('Input vertex counts missing; supply --observations')
        return
    observed = defaultdict(list)
    for line in observations_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('engine') in ENGINES:
            key = (row['spec']['family'], int(row['spec']['size']), int(row['spec']['seed']))
            observed[key].append(row)
    from statistics import median
    for row in rows:
        trials = observed[(row['family'], int(row['size']), int(row['seed']))]
        if trials:
            vertices = {t['input']['vertices'] for t in trials}
            if len(vertices) != 1:
                raise ValueError(f'Input vertex counts changed: {row}')
            row['vertices'] = str(vertices.pop())
            row['edges'] = str(trials[0]['input']['edges'])
        if row.get('outcome') != 'complete':
            continue
        if len({t['input_hash'] for t in trials}) != 1:
            raise ValueError('Engines did not receive the same input')
        for engine in ENGINES:
            measured = [t for t in trials if t['engine'] == engine and not t['warmup']]
            if len(measured) != 3 or {t['repetition'] for t in measured} != {1, 2, 3}:
                raise ValueError(f'Incomplete repetitions: {row["family"]}, {row["size"]}, {engine}')
            if any(t['status'] in ('failed', 'error') for t in measured):
                raise ValueError('Failed trial labeled complete')
            for scope in ('rewrite', 'total'):
                if not math.isclose(median(t[scope+'_s'] for t in measured), float(row[engine+'_'+scope]), rel_tol=1e-12):
                    raise ValueError('Published median differs from raw measurements')


def figure(rows, families, scope, kind, speedup=False):
    columns = 3 if kind == 'circuits' else 4
    width, height = ('.27', '.22') if columns == 3 else ('.195', '.17')
    ylabel = 'PyZX / database' if speedup else 'Time (s)'
    lines = [r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=' +
             f'{columns} by {math.ceil(len(families)/columns)},' +
             r'horizontal sep=1.5cm,vertical sep=2cm},',
             f'width={width}\\textwidth,height={height}\\textwidth,',
             r'xmode=log,ymode=log,log basis x=2,',
             r'tick label style={font=\scriptsize},label style={font=\scriptsize},',
             r'title style={font=\small},grid=major,grid style={black!10},',
             r'axis lines=left,legend columns=-1,',
             r'legend style={font=\scriptsize,draw=none,at={(0,1.5)},anchor=south west},',
             r'unbounded coords=jump]']
    engines = list(ENGINES)[1:] if speedup else list(ENGINES)
    for index, family in enumerate(families):
        selected = [r for r in rows if r['family'] == family]
        grouped = defaultdict(list)
        for row in selected:
            grouped[int(row['size'])].append(row)
        xlabel = {'circuits': 'Qubits', 'copies': 'Disjoint copies',
                  'rules': 'Input vertices'}[kind]
        title = NAMES[family.removeprefix('copies_')]
        ticks = ''
        if kind == 'circuits':
            tick_values = ','.join(str(size) for size in sorted(grouped))
            ticks = f',xtick={{{tick_values}}},xticklabels={{{tick_values}}}'
        lines.append(f'\\nextgroupplot[title={{{title}}},xlabel={{{xlabel}}},ylabel={{{ylabel}}}{ticks}]')
        xs = []
        for engine in engines:
            label, color, marker, style = ENGINES[engine]
            coordinates = []
            for size, group in sorted(grouped.items()):
                # Rule panels use actual graph size, not a synthetic "qubit" label.
                x = int(group[0]['vertices']) if kind == 'rules' else size
                if kind == 'rules' and len({r['vertices'] for r in group}) != 1:
                    raise ValueError(f'Inconsistent vertex counts: {family}, {size}')
                xs.append(x)
                summary = summarize(group, engine, scope, speedup)
                if summary is None:
                    coordinates.append(f'({x},nan)')
                    print(f'Omitted incomplete/invalid cell: {family}, {size}, {engine}, {scope}')
                else:
                    center, low, high = summary
                    coordinates.append(f'({x},{center:.9g}) += (0,{high-center:.9g}) -= (0,{center-low:.9g})')
            lines.append(r'\addplot+[' + f'color={color},mark={marker},{style},' +
                         r'line width=.7pt,mark size=1.5pt,error bars/.cd,y dir=both,y explicit] coordinates {' +
                         '\n' + '\n'.join(coordinates) + '\n};')
            if index == 0:
                lines.append(f'\\addlegendentry{{{label}}}')
        if speedup:
            lines.append(r'\addplot[gray,densely dotted,no marks,forget plot] coordinates {' +
                         f'({min(xs)},1) ({max(xs)},1)' + '};')
    return '\n'.join(lines + [r'\end{groupplot}', r'\end{tikzpicture}', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, help='Default: paper-figures beside the CSV')
    parser.add_argument('--observations', type=Path, help='Default: observations.jsonl beside the CSV')
    args = parser.parse_args()
    with args.csv.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('The timing CSV is empty')
    plan_path = args.csv.parent/'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        expected = [(j['spec']['family'], int(j['spec']['size']), int(j['spec']['seed'])) for j in plan['jobs']]
        actual = [(r['family'], int(r['size']), int(r['seed'])) for r in rows]
        if sorted(expected) != sorted(actual):
            parser.error('CSV configurations differ from the saved plan; missing or duplicate seeds cannot be silently plotted')
    observations_path = args.observations or args.csv.parent/'observations.jsonl'
    enrich_and_check(rows, observations_path)
    planned_count = len(rows)
    rows, coverage = select_completed(rows)
    if not rows:
        parser.error('No complete paired timing groups')
    present = {r['family'] for r in rows}
    known = set(NAMES) | {'copies_' + f for f in NAMES}
    if present - known:
        parser.error(f'Unknown families: {sorted(present - known)}')
    groups = {
        'rules': [f for f in NAMES if f in present and not f.startswith(('random_', 'layered_'))],
        'copies': ['copies_' + f for f in NAMES if 'copies_' + f in present],
        'circuits': [f for f in NAMES if f in present and f.startswith(('random_', 'layered_'))],
    }
    output = args.output or args.csv.parent / 'paper-figures'
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in [('plotted-data.csv', rows), ('coverage.csv', coverage)]:
        with (output/filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for r in data for k in r}))
            writer.writeheader(); writer.writerows(data)
    manifest = dict(input_csv=str(args.csv.resolve()),
                    input_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                    observations_sha256=hashlib.sha256(observations_path.read_bytes()).hexdigest() if observations_path.exists() else None,
                    planned_configurations=planned_count, plotted_configurations=len(rows),
                    selection='Contiguous complete size groups, paired across all three engines and all planned seeds; no extrapolation',
                    semantics='Timing completeness is not a semantic certificate; large-output checks remain inconclusive',
                    endpoints={f:max(int(r['size']) for r in rows if r['family']==f) for f in sorted(present)})
    (output/'plot-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    document = [r'\documentclass{article}', r'\usepackage[a4paper,margin=1cm]{geometry}',
                r'\usepackage{pgfplots}', r'\usepgfplotslibrary{groupplots}',
                r'\pgfplotsset{compat=1.18}', r'\pagestyle{empty}', r'\begin{document}']
    for kind, families in groups.items():
        if not families:
            continue
        for scope in ('rewrite', 'total'):
            for speedup in ((False, True) if kind == 'circuits' else (False,)):
                name = f'{kind}-{scope}' + ('-speedup' if speedup else '')
                path = output / (name + '.tex')
                path.write_text(figure(rows, families, scope, kind, speedup), encoding='utf-8')
                standalone = [r'\documentclass{article}',
                              r'\usepackage[a4paper,margin=1cm]{geometry}',
                              r'\usepackage{pgfplots}', r'\usepgfplotslibrary{groupplots}',
                              r'\pgfplotsset{compat=1.18}',
                              r'\usepackage[active,tightpage]{preview}',
                              r'\PreviewEnvironment{tikzpicture}', r'\setlength\PreviewBorder{5pt}',
                              r'\begin{document}', f'\\input{{{name}.tex}}', r'\end{document}', '']
                (output/(name+'-standalone.tex')).write_text('\n'.join(standalone), encoding='utf-8')
                print(path)
                caption = ('Geometric mean of configuration medians; bars span their min--max, '
                           'not confidence intervals or repetition ranges. '
                           'Curves end at the last contiguous complete three-engine size group; '
                           'all planned seeds must complete. No extrapolation. '
                           'Panels have independent axis limits. ')
                if kind == 'circuits':
                    caption += ('Local-layered seed labels repeat the same input. '
                                'Random depth = 20 times qubits; layered depth = 2. ')
                caption += ('Ratios above 1 favor the database. ' if speedup else '')
                caption += ('Rewrite excludes import/export.' if scope == 'rewrite' else
                            'Total includes in-memory import/export (PyZX: copying), not startup or disk I/O.')
                caption += (' Three measured repetitions per configuration. '
                            'Large-output semantic checks remain inconclusive; these are completed timing comparisons, '
                            'not correctness certificates. Runtime cutoffs changed during the campaign.')
                document += [r'\begin{center}', f'\\textbf{{{name}}}\\par\\medskip',
                             f'\\input{{{name}.tex}}', r'\end{center}', caption, r'\clearpage']
    document += [r'\end{document}', '']
    (output / 'main.tex').write_text('\n'.join(document), encoding='utf-8')
    print(f'{len(rows)} configurations; compile {output / "main.tex"}')


if __name__ == '__main__':
    main()
