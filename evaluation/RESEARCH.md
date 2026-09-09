# Reproducible rule and full-reduction studies

This suite evaluates the **current corrected implementation**, with a fixed,
inspectable study plan. It does not search for a favorable crossover. No large
campaign was run while implementing it. Existing regression tests remain separate.

## Run it

The reduction variants and their regression evidence are described in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md). Use `plan --db-variant optimized` to enable
independent pivot/lcomp batches and exclusive-call scan reductions. `worklist`
additionally enables indexed candidate frontiers. The default remains `control`;
use separate result folders for each variant.

From the repository root, with the project environment and a dedicated, empty
Memgraph instance:

```powershell
.\myenv\Scripts\python.exe -m evaluation.research plan --profile smoke --output smoke-plan.json
.\myenv\Scripts\python.exe -m evaluation.research run --plan smoke-plan.json --output results-smoke

.\myenv\Scripts\python.exe -m evaluation.research plan --profile study --output study-plan.json
.\myenv\Scripts\python.exe -m evaluation.research run --plan study-plan.json --output results-study --budget-seconds 3600
```

Inspect/edit the JSON plan **before** starting. `smoke` has 22 small workload
blocks, one warmup and two paired repetitions. `study` has 268 blocks: 17
individual-rule families at four scales and five circuit families at four widths,
two depths and five seeds, with one warmup and five paired repetitions per block.
These are starting grids, not a claim that one hour completes every cell.

For a larger pattern-count sweep, without accidentally creating huge circuits:

```powershell
.\myenv\Scripts\python.exe -m evaluation.research plan --profile study --families copies_lcomp copies_pivot copies_pivot_gadget --sizes 2 8 32 128 512 2048 --output patterns-plan.json
.\myenv\Scripts\python.exe -m evaluation.research run --plan patterns-plan.json --output results-patterns --budget-seconds 3600
```

Connection settings are `ZXDB_URI`, `ZXDB_USER`, `ZXDB_PASSWORD`; credentials are
not written to reports. Defaults are unauthenticated `bolt://localhost:7687`.
The runner refuses a nonempty database before constructing ZXdb, uses unique
graph IDs and scoped cleanup, and never calls the historical global importer or
database wipe. ZXdb still requires analytical storage mode; run against a dedicated
server without competing clients.

## Corrected baseline and massive inputs

Use a new plan and results folder for an ascending, time-bounded study:

```powershell
.\myenv\Scripts\python.exe -m evaluation.research plan --profile adaptive --db-variant optimized --output adaptive-plan.json
.\myenv\Scripts\python.exe -m evaluation.research run --plan adaptive-plan.json --output results-adaptive --restart-container memgraph --budget-seconds 3600
```

Repeat the run command to resume in the same folder. Only use `--restart-container`
for the dedicated benchmark database. New `massive` plans are an alias for this
adaptive profile. Existing plan files and results are not rewritten.

Every run now defaults to **120 worker seconds per case across retries**, including
generation, loading, reduction, validation and cleanup. This also caps old plans
that specified two hours. `--case-timeout-seconds` can lower this cap (above 10);
the effective cap is the smaller of this option and the plan's `block_timeout_s`.
The campaign budget must have at least one full case budget remaining to start a
new case. Database recovery has its own bounded time allowance, outside worker
time; reporting and initial preflight also add overhead.

Adaptive plans use **15 seconds per engine attempt**, with a 10-second database
query cap, one warmup and two measured repetitions. Database attempt time includes
loading and reading; PyZX uses a fresh subprocess, including startup/imports,
copying and IPC, with a sampled 4096 MiB RSS limit. Only reduction itself is recorded
as `reduce_s`. These setup protocols differ, so timeouts measure the declared
execution protocol, not a pure reduction-time lower bound. Neither the RSS limit
nor BLAS thread settings constrain Memgraph's memory or CPU allocation.

Within each workload series, sizes double from small inputs. Circuit depth grows
from 4 to 32768 at fixed widths 2, 8 and 32, with three seeds per depth. Independent
rule patterns grow from 2 to 131072; fan/gadget sizes stop at 512 because rewrites
can produce quadratic edge growth. Series are interleaved; their order and seed
order within a scale are deterministic from the saved random seed. Full reduction
uses realistic circuits without planted patterns. A small motif certifies rule
applicability outside timing. Input vertex/edge caps (2000000/4000000) are checked
after construction; generation is also subject to the hard worker timeout.

**Stop a series after three consecutive cases where both engines time out.**
A PyZX-only timeout does not stop database growth, and vice versa. Once an engine
times out, its remaining attempts within that case are marked
`not_run_after_timeout`; the other engine continues. If both time out, remaining
repetitions are omitted and a `limit_stop` record preserves that evidence, even if
it happened during warmup. The three cases can be different seeds at the same
circuit depth. Success breaks the timeout streak. Finite grid and campaign limits
can end a study before a timeout boundary is reached.

Worker crashes and exhausted whole-case budgets do not establish a both-engine
performance limit. A whole-case budget failure blocks further growth of that
series, with a distinct `case_budget` / `predecessor_case_budget` reason. Recovery
is required after a killed worker to avoid leaving a database query running.
Without recovery enabled, worker failure stops the campaign. Reports retain
`limit_stop_cases`, `adaptive_skip_cases` and `case_budget_cases`; skipped cells
never receive invented timings or speedups. Resume reconstructs stopping decisions
from saved observations. Adaptive campaigns require identical code, packages,
plan and time cap; use a new folder when changing them.

The fixed smoke/study profiles keep their randomized grids and in-process PyZX
protocol, but receive the default whole-case cap too. To get ascending growth,
generate a new adaptive plan. Keep execution protocols separate in comparisons.

A PyZX resource limit leaves its reduction time missing and allows the database
attempt and following repetitions to run. It does not restart Memgraph. Ordinary
database failures still use the existing recovery policy. `summary.csv` records
timeout/memory-limit counts and per-engine completed-attempt medians. Additional
`figures/*-capacity.tikz` fragments show those available timings, including unpaired
results. These are survivor-only summaries: consult the counts, and do not treat
them as unbiased paired comparisons. Normal timing/speedup plots still suppress
incomplete cells. No finite speedup is assigned to a timed-out baseline.

Record Docker CPU/RAM allocation and host RAM in `machine-notes.md`. For a fair
capacity claim, give both engines comparable declared CPU/RAM/time budgets; the
default PyZX child limits do **not** constrain Memgraph or total host memory.
Graph growth, parent-side generation/serialization and server allocation may
require substantially more RAM. Large tensor checks remain inconclusive beyond
their existing bound; graph/degree evidence alone is not a proof of equivalence.

Projective tensor validation now removes provably nonzero global scalar factors
before contraction, avoiding false zero-map failures from tiny PyZX scalars.
Actual zero scalars remain significant; unknown or numerically ambiguous scalar
factors yield inconclusive evidence.

Each block runs in a separate process with a wall-clock timeout. Both engines get
one-thread BLAS settings; this **does not constrain Memgraph's CPU allocation**.
A hard timeout or worker error stops the campaign by default. Opt-in automatic
recovery and checkpoint resume are described below. Completed observations survive;
budget limits record unrun cells. Runs automatically resume a recognized campaign
folder or initialize a new/empty folder; unrelated nonempty folders are refused.

### Automatic restart and resume

For a **dedicated local Docker container** named `memgraph`:

```powershell
# The same command starts a new campaign or resumes an existing one:
python -m evaluation.research run --plan study-plan.json --output results-study --restart-container memgraph --max-retries 1 --budget-seconds 3600

# After a crash, interruption or exhausted budget, run that command again.
```

Use the original `study-plan.json` so the command also works before `results-study`
exists. Generate it with the `plan` command first if needed. `--resume` remains
accepted for compatibility, but is optional and also works on the first run.

The runner verifies that the selected container publishes the port in `ZXDB_URI`.
Docker must be accessible from the terminal. On a server disconnect it restarts
that container and waits for a successful Bolt query (up to 90 seconds after
restart). A worker timeout also triggers restart, to stop any abandoned server
query. An ordinary worker error with a responsive server only needs scoped cleanup.
Unrelated database contents block cleanup; it deletes only exact graph IDs recorded
by this campaign. No global wipe is performed. Do not share this container with
another active campaign or application: a restart interrupts all its connections.

Completed paired trials are retained byte-for-byte and skipped. An interrupted
pair is rerun from its original input; completed results for only one engine within
that pair were not checkpointed. Both engines receive fresh warmups after recovery.
`--max-retries 1` allows one retry per case across resumed sessions. A repeatedly
failing case is marked `failed_retries_exhausted`, and evaluation proceeds to the
next case. Increase the limit explicitly to revisit exhausted cases. Docker or
readiness failures stop the campaign rather than causing an infinite restart loop.

`recovery.jsonl` records restarts; `sessions.jsonl` records each invocation's
environment and settings. Block logs append rather than overwrite. A truncated
last JSONL record is backed up before repair. An OS file lock prevents two runners
from writing the same result folder concurrently. Each resumed invocation has its
own time budget; it does not rerun completed cases to fill that budget.

Resume requires exactly the saved plan and matching source/package versions.
**For folders created before this recovery feature or after any code changes**, use:

```powershell
python -m evaluation.research run --plan study-plan.json --output results-study --restart-container memgraph --allow-code-change --budget-seconds 3600
```

This explicit override records a new version without relabeling old observations.
Changed input hashes are still rejected. Summary cells containing mixed versions
or any recorded failed attempt have their aggregate speedup suppressed; successful
raw timings remain available. Recovery is operational continuity, not evidence
that crashes disappeared. For clean before/after performance comparisons, use
separate campaigns with a fixed implementation version.

## Workload design and applicability

Individual-rule workloads construct known target patterns. Before timing, every
input is copied and reduced by its **named reference rule** to confirm actual
applicability. Constructed witness counts are recorded separately from progress;
they are not advertised as exhaustive matcher counts. Every measured database
result must change structural/phase-count metrics or the trial is `no_progress`,
which prevents an aggregate speedup from being reported for that cell. This is a
conservative progress guard, not a proof that every possible match was applied.

The `copies_*` families tensor together independent, fixed-size rule motifs.
Increasing size increases guaranteed opportunities without changing each motif's
neighborhood. The base families vary neighborhood size or pattern count. Comparing
the two separates scaling with the number of opportunities from scaling with the
cost of an individual rewrite. `identity_chain` adds overlapping opportunities.

| Paper Figure 16 panel | Current base family | Scale | Additional controlled family |
|---|---|---|---|
| (a) Pivot | `pivot` | Exclusive neighbors per endpoint | `copies_pivot` |
| (b) Pivot gadget | `pivot_gadget` | Neighbors per endpoint | `copies_pivot_gadget` |
| (c) Local complementation | `lcomp` | Center degree | `copies_lcomp` |
| (d) Phase gadget | `gadget_fusion` | Gadgets sharing two targets | `copies_gadget_fusion` |
| (e) Spider fusion | `spider_fusion` | Independent fusable pairs | `copies_spider_fusion` |
| (f) Identity | `identity` | Independent identities | `copies_identity`, `identity_chain` |
| (g) Bialgebra | `bialgebra` | Neighbors per endpoint | `copies_bialgebra` |
| (h) Pivot boundary | `pivot_boundary` | Neighbors per endpoint | `copies_pivot_boundary` |

Actual vertices, edges, I/O counts, maximum/mean degree and non-Clifford spider
count are always recorded. An x-coordinate called `fan_size` is a construction
parameter, not a qubit count or necessarily the final degree.

The historical scripts mix random circuits, repeated fixtures and changing graph
degrees; some pivot-gadget graphs have no applicable match because their neighbors
have degree one. This suite attaches boundary legs to these fan leaves and exposes
gadget-fusion targets as an open map. The fixtures are intentional adaptations,
not exact historical datasets. An exact numerical reproduction of Figure 16 would
also require its original seeds, repetitions, hardware limits, query version and
complete parameter grids. Keep historical results and these corrected studies
distinct in the paper. The suite covers the eight Figure 16 rule families, not
every additional ZXdb rule; existing rule regression evaluations cover the rest.

### Full reduction: realistic inputs, no planting

* `random_clifford`: generated CNOT/H/Clifford circuits.
* `random_clifford_t`: CNOT/H/T circuits with generator `p_t=0.25`.
* `random_t_heavy`: the same generator with `p_t=0.60`.
* `layered_local`: nearest-neighbor CNOT–ZPhase–CNOT interactions and X mixers.
* `layered_global`: uniformly sampled interaction pairs with the **same edge
  budget per layer** as the local circuit.

Widths and depths are varied separately. `depth` means generator gate depth for
random circuits and interaction layers for layered circuits; never pool these
axes. An infeasible locality/edge-count request raises an error instead of silently
reducing density. The realistic inputs contain no planted rewrite motifs, and
are allowed to have different rule frequencies. Their correctness, reduction time
and output size are evaluated as whole algorithms, not as evidence about the
frequency or cost of any specific rule.

## Measurement and analysis

* Generation, DB loading / PyZX copying, reduction, DB reading and semantic
  validation have separate timers. The primary comparison is `reduce_s` after
  setup. Load/read costs remain in raw data; add them when studying a workflow
  that requires transfers. This benchmark alone is not an application throughput
  or end-to-end speedup claim.
* The input is freshly loaded/copied for **every** warmup and repetition. Both
  engines run on the same input. AB/BA order alternates with a seeded random
  starting order, and block order is seeded and shuffled. Warmups are excluded.
* Baseline contract `pyzx-0.10.3-native-v2` requires PyZX 0.10.3. Time one native
  call for `full_reduce`, spider fusion, identity, local complementation, pivot,
  and bialgebra: these APIs already implement their iteration internally.
  Standalone pivot-gadget, boundary-pivot, and gadget-fusion APIs apply one batch,
  so repeat their returned change flag until false. No graph serialization or
  topology fingerprint is charged to reduction. Earlier results used a redundant
  outer fixed-point adapter and must be rerun in a **new output directory**;
  do not pool them with this corrected baseline.
* The query counter wraps public session and managed-transaction calls. It counts
  submitted Cypher statements during reduction, not transport round trips or
  server operators. Warm caches persist within a block; server caches can persist
  across blocks. This is a warm steady-state protocol, not a cold-start study.
* Speedup is **PyZX time / ZXdb time**. Each input contributes its median paired
  log-ratio; the reported point is their geometric mean. A seeded hierarchical
  bootstrap resamples distinct input hashes, then repetitions within inputs,
  giving a 95% percentile interval. Identical inputs from different seeds are
  merged, so they do not become fictitious independent circuits.
* Single-input intervals describe timing variability only. Five random seeds
  are an initial study, not a universal adequacy guarantee. For publication, use
  more independent inputs, multiple sessions and machines, and inspect interval
  stability. A speedup interval crossing 1 does not establish which is faster.
  There is no multiple-comparison correction; do not cherry-pick significant cells.
* Any failed, no-progress or missing measured trial, failed attempt, failed warmup,
  or mixture of source/package versions suppresses that cell's
  aggregate times and speedup. Its raw observations and failure counts remain.
  Semantically inconclusive but otherwise complete cells have timings, with an
  explicit `inconclusive` evidence label.

This separation of input variability and repeated timing noise follows the
experimental concerns in [Kalibera and Jones, Rigorous Benchmarking in Reasonable
Time](https://kar.kent.ac.uk/33611/). The bootstrap here is the explicitly described
paired estimator, not an implementation of every method in that paper.

## Correctness evidence and limits

Every warmup and measured trial also compares the **ZXdb output directly with
the PyZX output** at three levels, outside reduction timing:

1. **Tensor:** ordered tensor equality up to nonzero global scalar when feasible.
   This can succeed even when the original input was too large for verification,
   if both reduced outputs are small. Agreement with PyZX alone does not replace
   checking the original circuit.
2. **Graph:** exact multigraph isomorphism, preserving spider colors, phases modulo
   2, ordered input/output roles, wire types, parallel-edge multiplicities and
   self-loops. Scalars and drawing coordinates are excluded. Exact search has a
   256-vertex / 4096-edge cap and a cooperative one-second search deadline;
   over-budget checks are inconclusive. Degree or edge-count differences can
   establish non-isomorphism without a search at any size.
3. **Degree sequence:** equality of degree histograms, including multiplicities,
   for all sizes. Both histograms are saved in raw observations.

Raw trial records contain `comparison.tensor`, `comparison.graph`,
`comparison.degree_sequence` and the separate `comparison_s` timer. Summary CSV/JSON
columns beginning `outputs_` count each outcome. Older campaigns are labeled
`not_recorded`; regenerating their report does not invent validation results.
A direct tensor mismatch marks the trial failed and saves both outputs. Graph or
degree disagreement remains visible but does **not** imply a semantic error:
different valid reduction orders can produce different equivalent graphs. Likewise,
degree agreement never counts as semantic verification.

For feasible cases, both outputs are compared **independently against the original
input**, using ordered dense tensors up to a nonzero global scalar. There is no
boundary-permutation search or fallback that treats degree agreement as semantic
success. Zero maps are checked explicitly. Invalid boundaries fail. Multiedges
and self-loops survive direct database reading. The original and reference PyZX
scalars are retained; ZXdb does not track general scalar factors, so exact scalar
equality is outside the supported contract.

Dense verification is conservatively limited to 32 vertices and an edge-plus-open-
leg budget of 20, bounding intermediate tensor allocation. Large inputs/results
are **inconclusive**, not passed. Saved graph payloads allow later independent
verification. The current suite does not prove correctness of large circuits;
small strict semantic regressions and large performance measurements are separate
forms of evidence. Floating-point comparisons use a residual threshold of 1e-8;
near-zero values remain a numerical limitation.

## Artifacts and paper figures

Each campaign writes its plan, package/platform metadata and source hashes
(including uncommitted code), per-block server information, exact input payloads,
first measured outputs, failure outputs and flushed raw JSONL observations.
`environment.json` does not measure Memgraph's container limits or physical RAM;
add a `machine-notes.md` with CPU model, RAM, container CPU/memory allocation,
server flags, concurrent load and OS power mode before publication.

```powershell
# Rebuild tables/figures without touching Memgraph:
.\myenv\Scripts\python.exe -m evaluation.research report results-study
cd results-study\figures
pdflatex figures.tex
pdflatex paper_rules.tex
```

`summary.csv` and `summary.json` retain evidence labels and sample counts.
`figures.tex` contains timing and paired-speedup plots for every family/depth;
`paper_rules.tex` arranges the eight base rules in the paper's 4-by-2 panel order.
Each editable `.tikz` fragment reads an accompanying plain-text `.dat` table.
PGFPlots handles log axes and asymmetric speedup intervals natively; no
`tikzplotlib` dependency is needed. Run LaTeX in the figures directory, or configure
its table search path when embedding fragments in your paper. Requires PGFPlots
1.18, and the `standalone` class for `paper_rules.tex`.

Curves include inconclusive large cases; captions must say so. Failed/missing cells
are `nan` and break curves. Absolute-runtime curves show medians; uncertainty bars
are on the paired speedup plots. The eight-panel graphic uses blue squares for
ZXdb and red circles for PyZX; explain that in the caption.

## Validate the suite

```powershell
.\myenv\Scripts\python.exe -m unittest tests.test_research tests.test_research_adaptive tests.test_research_recovery -v
# Optional live correctness and two-input pipeline checks, not a scale campaign:
$env:ZXDB_RESEARCH_LIVE='1'
.\myenv\Scripts\python.exe -m unittest tests.test_research tests.test_research_adaptive tests.test_research_recovery -v
Remove-Item Env:ZXDB_RESEARCH_LIVE
```

The tests check reproducibility, applicability at multiple scales, eight rule
motifs, realistic full reduction, invalid locality requests, zero-map/incorrect
phase detection, topology fingerprints, both query submission paths, hierarchical
aggregation, warmup exclusion, failure visibility and PGFPlots artifact generation.
