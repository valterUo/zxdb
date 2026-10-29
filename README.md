# ZXdb

Reproduce the October 29, 2026 comparison of PyZX, Cypher rewrites in Memgraph,
and hybrid Cypher/C++ rewrites. The [final report](results/generous-rerun-20261029/finalized/REPORT.md)
covers 210 completed configurations and 1,890 measured engine trials.
The original 257-configuration campaign was stopped early; incomplete outcomes
remain recorded and are excluded from paired timing medians.

## Setup

Use Python 3.12 and install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
```

Benchmarks require Docker and a dedicated empty Memgraph 3.14.0 instance,
with no concurrent writers. Create one (PowerShell):

```powershell
docker run --detach --name zxdb-reproduce --publish 7692:7687 memgraph/memgraph:3.14.0
$env:ZXDB_URI = "bolt://localhost:7692"
```

Build and load the native module using [these instructions](zxdb_qm/README.md),
substituting `zxdb-reproduce` for the container name. All three engines are
needed for the published comparison. The original build details are in
[build.json](results/memgraph-3.14/build.json); campaign environment and resource
records are under `results/generous-rerun-20261029/`.

## Reproduce the benchmarks

Run the correctness gate first, then replay the exact 210 completed workload
configurations from the saved outcomes. Choose fresh output directories.

```powershell
python evaluate.py --correctness-only --variants optimized native --regression-timeout 7200 --trial-timeout 7200 --restart-container zxdb-reproduce --output results/reproduced/validation
```

Continue only if the correctness command succeeds:

```powershell
python -c "import json,pathlib; p=pathlib.Path('results/reproduced'); rows=json.loads(pathlib.Path('results/generous-rerun-20261029/finalized/outcomes.json').read_text()); (p/'manifest.json').write_text(json.dumps([dict(rule=r['rule'],spec=r['spec']) for r in rows if r['outcome']=='complete']))"
python -m evaluation.rerun_three_engines --container zxdb-reproduce --manifest results/reproduced/manifest.json --engine-timeout 7200 --output results/reproduced/performance
python -m evaluation.report_three_engines results/reproduced/performance
```

Each engine gets one warmup and three measured repetitions. Random circuit depth
is 20 times qubit count; layered circuits use two layers. Timings depend on the
machine and cannot be expected to match exactly. The original campaign's changing
cutoffs, skipped cases, and failures remain in its plan, policies and outcomes;
this replay intentionally benchmarks only its completed configurations.
`python -m evaluation.run_generous_campaign --help` describes the original full-grid
supervisor if all 257 planned configurations are wanted.

## Reproduce the figures and summary

The paper figure generator checks the frozen timings against raw observations
and needs only Python's standard library:

```sh
python evaluation/plot_timings.py --output results/reproduced/paper-figures
```

Compile `main.tex` in that output directory with `pdflatex` (PGFPlots required).
For individual cropped PDFs, compile each `*-standalone.tex` using the figure
name as `-jobname`. The [saved preview](results/generous-rerun-20261029/finalized/paper-figures/main.pdf)
and [figure notes](results/generous-rerun-20261029/finalized/paper-figures/README.md)
include captions and interpretation. Pass another `timings.csv` as the first
argument to plot a fresh run.

To rebuild the frozen summary, CSVs and three PNG/SVG report figures from the
retained per-case evidence, use a fresh destination:

```sh
python results/generous-rerun-20261029/finalize_results.py --output results/reproduced/finalized
```

Speedup is PyZX time divided by database time. Total time includes graph
import/rewrite/export. Plot ranges span configuration medians, not confidence
intervals. Large-output semantic checks remain mostly inconclusive; completed
timings do not establish correctness. Local-layered seed labels repeat inputs,
and general scalar tracking is incomplete.

## Repository contents

- `zxdb/`, `zxdb_qm/`: measured rewrite implementation and native build instructions.
- `evaluate.py`, `evaluation/`: fixtures, generators, runners, verification and figures.
- `tests/`: regression checks (`python -m unittest discover -s tests -v`;
  database tests are opt-in and also run through the correctness gate).
- `results/generous-rerun-20261029/`: raw cases, graph snapshots, source archives,
  correctness evidence, policies, and finalized results.

Older campaigns and development material were moved to the sibling backup
`../zxdb-backup-20261026/cleanup-20261029/`, with a SHA-256 archive manifest.
They are not needed for the commands above. Archived historical helper sources
remain available in the retained measured-source ZIPs for provenance checks.
