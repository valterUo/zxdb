"""Join distinct measured sizes for plotting, without pooling repetitions."""
import csv
import re
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCES = [REPO / 'results/rerun-20261028/performance/timings.csv',
           REPO / 'results/rule-scaling-20261028/timings.csv', HERE / 'timings.csv']


def main():
    rows, seen = [], set()
    for path in SOURCES:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['family'].startswith(('random_', 'layered_')):
                    continue
                if not all(row[e + '_complete'] == 'True' for e in ('pyzx', 'cypher', 'hybrid')):
                    continue
                key = row['family'], row['size'], row['seed']
                if key in seen:
                    raise ValueError(f'Duplicate configuration across campaigns: {key}')
                seen.add(key)
                row['campaign_csv'] = path.relative_to(REPO).as_posix()
                rows.append(row)
    joined = HERE / 'combined-timings.csv'
    with joined.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    output = HERE / 'figures'
    subprocess.run([sys.executable, str(REPO / 'evaluation/plot_timings.py'), str(joined),
                    '--output', str(output)], check=True)
    for path in output.glob('*.tex'):
        text = path.read_text()
        if path.name == 'main.tex':
            text = text.replace('Geometric mean of configuration medians;',
                'Original, extension and retry campaigns; unchanged rewrite implementations. '
                'Each point uses one campaign; repetitions are not pooled. '
                'Failed or unrun larger cases are listed in the accompanying report. '
                'Geometric mean of configuration medians;')
        else:
            text = text.replace('log basis x=2,',
                'log basis x=2,\n' + r'x tick label style={rotate=90,anchor=east},')
            panels = text.split(r'\nextgroupplot[')
            for i in range(1, len(panels)):
                sizes = sorted({int(x) for x in re.findall(r'^\((\d+),', panels[i], re.M)})
                labels = ','.join(map(str, sizes))
                panels[i] = f'xtick={{{labels}}},xticklabels={{{labels}}},' + panels[i]
            text = r'\nextgroupplot['.join(panels)
        path.write_text(text)
    print('Every plotted row records its campaign_csv; no duplicate sizes were pooled.')


if __name__ == '__main__':
    main()
