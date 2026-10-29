# ZXdb

A research prototype comparing ZX-calculus rewrites in Memgraph with PyZX.
The aim is a readable demonstration of when database batching helps, including
cases where database overhead makes it slower. This is not a production service.

The [finalized 2026-10-29 larger-budget rerun](results/generous-rerun-20261029/finalized/REPORT.md)
completed 210 configurations before the user stopped the campaign. At 128
qubits, Cypher wins all 15 configurations including import/export; hybrid wins
13/15. Larger standalone workloads show substantial gains for selected rules
and substantial losses for others. The report includes frozen measurements,
scaling figures, explicit incomplete-case outcomes, and correctness limitations.

The [2026-10-28 three-engine rerun](results/rerun-20261028/REPORT.md) covers all
28 performance workload shapes and 160 configurations, with three measured
repetitions. At 128 qubits, Cypher wins 14/15 circuit configurations on rewrite
time and hybrid Cypher+C++ wins 15/15; each wins 12/15 including import/export.
Standalone-rule results are mixed. All 223 correctness fixtures pass in both
database modes after supplemental checks, but some large measured outputs remain
semantically unresolved. The report includes full timings, scaling plots and
separate tensor, isomorphism and degree-sequence evidence.

All evaluation evidence is indexed in [results/README.md](results/README.md).
The [2026-10-27 rule audit](results/rule-audit-20261027/REPORT.md) adds
adversarial semantic tests and fixes discovered in current and retained rules.
The earlier Cypher-only timing campaign passed 65 regressions and won all 15
128-qubit configurations on median rewrite and total time (three measured
repetitions each). All 223 correctness fixtures are verified, including three
resolved by supplemental exact checks. All 37 distinct measured input/output
pairs have semantic certificates, covering all 45 measured trials. The final
regression suite passed 66 tests. Equivalence is up to a nonzero global scalar.
The earlier campaign with C++ procedures favored the database in all 15
128-qubit configurations; those results are separate from the Cypher evidence.
See the results index for versions, output checks and cases where PyZX wins.
The pre-cleanup repository is preserved in the independent sibling directory
`zxdb-backup-20261026`.

## Run the complete evaluation

Use Python 3.12 and a **dedicated empty Memgraph instance** on
`bolt://localhost:7687`. No concurrent database writers may run during evaluation.
The runner uses analytical storage mode, as required by the rewrite queries.
It removes obsolete type/worklist indexes from the empty evaluation instance;
rewrites retain indexes on stable node IDs and graph ownership. Mutable type
indexes caused crashes in Memgraph 3.6.2 during color-change stress tests.

```sh
python -m pip install -r requirements.txt
python evaluate.py
```

On this Windows checkout, use `myenv/Scripts/python.exe evaluate.py`.
Set `ZXDB_URI`, `ZXDB_USER`, and `ZXDB_PASSWORD` for a different connection.

One command runs implementation regressions, 223 correctness inputs against both
control and optimized database implementations, then a fixed scalability grid
against PyZX 0.10.3. It writes everything under `results/latest/`:

- `REPORT.md`: correctness, timings, failures and interpretation.
- `observations.jsonl`: every trial, including warmups and failed trials.
- `summary.json` and `summary.csv`: paired median speedups and observed ranges.
- `environment.json`: versions, machine details, configuration and source hashes.
- `regressions.log`: detailed regression results.
- `graphs/`: deduplicated graph snapshots for later correctness checks.
- `sources-attempt-*.zip`: exact measured source files for each attempt.

`python evaluate.py --quick --output results/quick-new` keeps **all correctness
inputs** but uses a smaller performance grid and one repetition. The full grid
uses three repetitions after one warmup. Choose a fresh output directory for a
new run; results are never silently overwritten. A failed check gives a nonzero
exit code after the report is saved. Inconclusive checks are reported separately.

For a dedicated local Docker container, `python evaluate.py --restart-container
memgraph --output results/full` can continue after a server crash. The failed
trial remains an error and its performance cell receives no speedup. This option
also enables a database watchdog (`--trial-timeout 120`, in seconds), covering
only database import/rewrite/export. Reference execution, snapshot writing and
verification stay outside that watchdog. After an
execution error, remaining repetitions of that cell are left missing. Without
this option, database loss stops the run and saves a partial report. Memgraph
3.6.2 crashed during exploratory full-reduction runs on this machine.

After an interrupted run, repeat the same command and settings with `--resume`.
Completed observations and failures are retained; failed cells are not retried.
Changing workload, oracle or rewrite sources prevents continuation. Runner
versions are snapshotted per attempt, and timing cells spanning attempts receive
no aggregated speedup. Regression execution is capped at five minutes.

For circuit benchmarks, `--certificate-timeout 120` optionally attempts a bounded
algebraic identity proof when tensor verification is inconclusive. This relies on
the input being unitary by circuit construction and is never applied to arbitrary
rule fixtures. Proofs run outside rewrite timing and the database watchdog, and
are cached by input/output hashes. A timeout remains inconclusive.

`--fixed-point-timeout 120` separately runs another PyZX `full_reduce` on each
unique circuit output, outside all timers. The report records whether its
structure changes; this is a reduction-completeness audit, not a semantic proof.
For simple outputs, the audit preserves vertex IDs, phases, boundary order and
scalar while visiting boundary/leaf edges first. This avoids expensive late
candidate rejection in PyZX; it does not change the timed reference execution.
Parallel edges and self-loops retain their original representation.
The runner also flags seed labels that generate identical inputs.

## What is measured

All eleven implemented rule families are included: spider fusion, identity and
Hadamard cancellation, local complementation, ordinary/gadget/boundary pivot,
phase-gadget fusion, bialgebra, copy, and supplementarity. `full_reduce` is tested
on adversarial compositions, random Clifford and Clifford+T circuits, T-heavy
circuits, QFT and layered circuits. Graph isolation, parallel-edge normalization,
and batching regressions are retained in `tests/`.

The performance grid separates disjoint pattern copies, growing neighborhoods,
and actual circuits. Each timed trial starts from the same input graph; engine
order alternates. PyZX runs its own rule functions, repeating only its three
single-batch APIs until completion. Its `full_reduce` is called once.

Speedup is **PyZX seconds / database seconds**: values above one favor ZXdb.
Rewrite-only and import/rewrite/export timings are reported separately. Import
and export use the small graph adapter in `evaluation/core.py`, not the legacy
JSON APIs or visualization layout. There is no claim that a synthetic rule win
implies faster full circuit reduction. Large tensors can exceed the oracle's
resource bounds; those measurements remain explicitly unverified. Near-zero or
nonfinite numerical tensors are inconclusive unless an exact zero factor gives
a decisive comparison. The saved report documents the conservative review of
evidence collected before this numerical-checker correction.

## Paper figures (TikZ/PGFPlots)

`python evaluation/plot_timings.py` reads the finalized 2026-10-29 campaign's
`timings.csv`, checks its complete-case medians against the raw observations,
and writes eight TikZ snippets, standalone wrappers, and a `main.tex` preview
under `results/generous-rerun-20261029/finalized/paper-figures/`. The compiled
preview and eight vector PDFs are included there. Python needs only the standard
library for generation. To select another campaign:

```sh
python evaluation/plot_timings.py path/to/timings.csv --output path/to/figures
```

Compile `main.tex` with `pdflatex` from its output directory. To include a snippet
in a paper, load `pgfplots`, `\usepgfplotslibrary{groupplots}`, and
`\pgfplotsset{compat=1.18}`, then use `\input{path/to/rules-rewrite.tex}` inside
a full-width `figure*`. Add the caption from the preview document. The snippets
scale with `\textwidth`; use a full-width float in two-column papers.

The small-multiple layout follows Figure 16 of arXiv:2511.13033v1, extended to
three engines and all 28 measured workload shapes. Separate grids cover growing
rule workloads (including identity chains), disjoint copies, and five circuit
families, each for rewrite and total time. Two further grids show circuit
speedups with a break-even line. Rule axes use input vertices, copies use motif
count, and circuits use qubits. These are new measurements, not reproductions
of the old paper's numerical curves or scale range.

Points are geometric means of the CSV's per-configuration timing medians.
Error bars are the minimum and maximum of those medians, **not confidence
intervals** or within-run timing uncertainty. A single configuration has no
visible range. Speedups aggregate per-configuration PyZX/database ratios.
Each curve ends at the first incomplete size group, requiring all three engines
and all planned seed configurations. All five full-reduce families reach 128
qubits; incomplete 256-qubit results are omitted. `coverage.csv` records every
inclusion and exclusion; `plot-manifest.json` records input hashes and endpoints.
No-op configurations are excluded from ratios
but remain in runtime plots. Timed completion does not establish semantic
correctness. The default campaign has three measured repetitions per
configuration, three seed labels per circuit size, and only one distinct
local-layered input per size. Historical campaigns are never pooled.

For a stronger scientific presentation:

- Put full-circuit total time and paired speedup in the main paper; retain
  rewrite-only and the complete rule grids as supporting mechanism evidence.
  Report crossover intervals between measured sizes rather than interpolated
  exact crossover claims.
- Show raw repetition points from `observations.jsonl` in a supplement. Keep
  timing variability separate from input variability. More independent graph
  instances are needed for persuasive population-level confidence intervals;
  repeated local-layered seed labels are not independent instances.
- Add import/rewrite/export breakdowns from raw trials, aggregating components
  consistently. Do not subtract separately computed medians to estimate median
  overhead. This can explain why rewrite wins disappear in total time.
- Relate runtime to input edges, degree and output graph size as well as qubits.
  Report simplification quality alongside runtime, because engines can follow
  different rewrite trajectories. Vertex reduction alone is not gate or T-count
  reduction; use extracted circuits if claiming compilation-quality gains.
- Pair performance with a separate semantic-verification coverage table using
  the finalized audits. Preserve inconclusive cases; degree agreement is not a
  correctness certificate. Do not infer asymptotic complexity from three rule
  sizes or from circuit families with different depth schedules.

## Reading the code

1. `evaluate.py`: the single evaluation entry point and report generation.
2. `evaluation/cases.py` and `circuits.py`: correctness fixtures.
3. `evaluation/patterns.py` and `workloads.py`: scalable deterministic inputs.
4. `evaluation/core.py`, `oracle.py` and `evidence.py`: graph I/O, verification,
   exact snapshots and bounded follow-up audits.
5. `zxdb/zxdb.py`: Python rewrite loops; `zxdb/main_queries.json`: Cypher rules.
6. `zxdb_qm/zxqmcpp.cpp`: scoped native rewrite loops.

The `optimized` variant uses **Cypher only**, with Python coordinating the rule
schedule. Interior and Pauli-boundary pivot cache eligible endpoints per query,
select disjoint neighborhoods greedily, and clear temporary labels before
rewriting. Interior pivot and local complementation prefer smaller neighborhoods
to reduce edge-toggle work. Correlated aggregates avoid repeatedly grouping by
whole neighbor lists. Edge toggles expand each source neighborhood once, and
supplementarity groups equal neighborhoods before comparing phases. No custom query module is
required. The completed five-family timing campaign and semantic audit
are indexed in the results README; the timings are workload-specific evidence.

To reproduce the repeated comparison on a fresh stock Memgraph 3.14.0 instance
(PowerShell; port 7690 must be unused):

```powershell
docker run --detach --name zxdb-research --publish 7690:7687 memgraph/memgraph:3.14.0
$env:ZXDB_URI = "bolt://localhost:7690"
python evaluate.py --variants optimized --full-reduce-only --qubits 128 --depth-factor 20 --seeds 3 --repetitions 3 --fixed-point-timeout 30 --output results/cypher
```

This still runs every correctness fixture before timing all five circuit families.
Add `--certificate-timeout 1800` to attempt large circuit identity certificates
inside the same pipeline. The hardest saved reference-based 128-qubit proofs
needed about 22 minutes; proof runtime depends on the comparison graphs and
shorter bounds can remain inconclusive even for equivalent outputs.
These checks run outside measured regions and can add several minutes per output.
Certificate subprocesses use `evaluation/proof_matching.py` to cache gadget
matcher adjacency and preserve PyZX's candidate order. Differential regressions
compare its matches and graph mutations with stock PyZX. The parent process and
timed PyZX baseline retain the stock matcher.
Use `--qubits 64 128` to include the smaller inputs where database overhead can
outweigh batching gains. No compiler or custom query module is needed.

`zxdb_qm/` provides optional native database procedures for interior pivot, local
complementation, boundary pivot, gadget pivot and grouped supplementarity. See its [build instructions](zxdb_qm/README.md).
Use the same evaluation entry point for larger circuit comparisons:

```sh
python evaluate.py --variants native --full-reduce-only --qubits 32 64 128 --depth-factor 20 --seeds 3 --output results/native
```

This still checks all correctness inputs before benchmarking. `--variants`
also accepts `control optimized native` for a direct three-way comparison.
Native procedures require compilation and a database restart before evaluation.
The default evaluation needs no compiled module.

Known limitations include incomplete scalar tracking and the legacy JSON
import/export APIs. Standalone Cypher supplementarity now checks its local
graph-like preconditions. The
evaluation adapter preserves explicit zero scalar factors as isolated pi spiders,
and cleanup retains those zero factors. General scalar tracking is still absent. See `evaluation/audit-history.md` for the prior audit, and the
new report for evidence on this revision. Historical campaign summaries are kept
in `results/historical-summary.json`; raw obsolete runs live in the backup repo.
