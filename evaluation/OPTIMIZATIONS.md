# Reduction optimizations

The original `full_reduce` remains the control. `full_reduce_optimized` follows
the same rule schedule, with the changes below. Matching order can differ, so
equivalent outputs need not have identical graphs or vertex counts.

## Implemented steps

1. The experimental batched-pivot selector now groups neighbourhood claims by
   vertex and then candidate. It no longer scans a global ownership list for each
   candidate. It selects an independent set with guaranteed progress, not
   necessarily a maximal set. Edge toggles materialize existing relationships
   before deletion, as in the corrected single-pivot query.
2. Local complementation has a corresponding independent-batch implementation.
   Pivot and local complementation batches require disjoint closed neighbourhoods.
   No arbitrary overlapping matches are rewritten together.
3. An explicit exclusive-access mode validates graph ownership once at the entry
   to a reduction. Ordinary standalone rules retain their guards. Options are
   invocation-local and reset even when an exception occurs.
4. Exclusive optimized calls remember successful parallel-edge normalization.
   Identity removal and fusion invalidate that state before normalizing new
   wiring. Other successful rewrites conservatively invalidate it; no-op calls
   preserve it. No cached graph state survives the call.
5. Optimized spider-fusion and supplementarity guards inspect at most one and two
   matching rows, respectively. This avoids counting all matches in positive
   cases; proving that no match exists can still require a scan.
6. Optional indexed worklists seed pivot/lcomp candidates once per fixed-point
   invocation. Each batch retains pending eligible candidates and marks affected
   neighbours for reconsideration. Other rules trigger a fresh seed rather than
   assuming a stale frontier remains valid. Reserved `_ZXdbPivotWork` and
   `_ZXdbLcompWork` labels are graph-scoped and cleanup is attempted on exit;
   stale markers from an interrupted call are discarded at the next entry. Worklists require
   exclusive access. They still inspect pending candidates to form each batch;
   this is not a native constant-time queue or an incremental implementation of
   every ZX rule.

Each step passed the existing 108 rule cases and 113 full-reduction cases before
proceeding. Final variant checks also route applicable standalone rules through
the optimized implementation. The historical harness reports 220 tensor matches
and one sampled-tensor match out of 221 cases; this is bounded empirical coverage,
not a proof for arbitrary graphs. New tests cover overlap, newly created
candidates, no-match behavior, foreign graphs, failure cleanup, submission counts
and the earlier Hadamard/parallel-edge counterexamples.

## Gadget-pivot batching (October 2026)

`apply_rule_optimized("pivot_gadget_rule", graph_id, exclusive_access=True)`
and `full_reduce_optimized` now use `Pivot gadget - batched`. The ordinary
`pivot_gadget_rule` retains the one-match query as a performance control.
Boundary-pivot batching is described below; bialgebra batching remains future work.

Each gadget-pivot candidate claims its two centers and all their neighbors.
Claims are grouped by vertex; the smallest candidate edge ID owns that vertex.
A candidate is selected only if it owns every vertex it claims. Selected
closed neighborhoods are disjoint, including shared-neighbor conflicts.
This is a non-maximal independent set with progress guaranteed by the minimum
ranked candidate. The query materializes the selection before mutation and
materializes existing edge matches before toggling them. The driver repeats
until no candidate remains. This batches work inside a query; it does not
introduce concurrent rewrite threads.

Batching alone was insufficient. `PROFILE` on 128 copies showed four neighbor
lookups performing `ScanAll` for each candidate (163,968 hits each), taking
most of the approximately 205 ms profiled statement. Omitting redundant
`:Node` labels on these bound-neighbor matches changed the plan to expansions
from the already-bound centers, reducing the profiled statement to about 14 ms.
The new regression checks that neighbor scans do not return. Planner behavior
is version-dependent; the measurement used Memgraph 3.6.2.

Integration testing also exposed a pre-existing problem in the single-Pauli
boundary query: its sorted candidate projection could carry incorrect boundary
and phase values on Memgraph 3.6.2. The initial unsorted/re-matched repair was
not sufficient under the later boundary test sequence. The single-match control
now materializes complete candidate tuples before taking one and materializes
edge deletions. Small circuit seeds 1 and 10 reproduce the old failure and are
tested repeatedly. This is a correctness repair, not a confirmed native engine
root-cause diagnosis.

### Bounded diagnostic results

Raw measurements: `audit_results/gadget-pivot-batch-probe.json`. Three distinct
motif mixtures per size (25%, 50%, 75% size-3 motifs, with the remainder size-2),
one excluded warmup and two measured repetitions per input. Engine order is
seeded and shuffled. Graph creation/copying, database loading, export and
validation are excluded. Both engines run gadget pivot to a fixed point.
PyZX is 0.10.3; the database ran in a dedicated local Docker container.

| Copies | Inputs | Measured pairs | Geometric mean PyZX / ZXdb |
|---:|---:|---:|---:|
| 32 | 3 | 6 | 0.35 |
| 64 | 3 | 6 | 0.52 |
| 128 | 3 | 6 | 0.94 |
| 256 | 3 | 6 | 1.54 |
| 512 | 3 | 6 | 3.86 |
| 1024 | 3 | 6 | 9.60 |

The estimator is `exp(mean(log(t_pyzx / t_zxdb)))` over measured pairs; each
input has equal weight because every included input has two repetitions.
Values above one favor ZXdb. These are exploratory diagnostics, not confidence
intervals or publication population estimates. The 180-second campaign budget
expired during the second 2048-copy case. Its first case completed (two measured
pairs, 15.66x), but is not pooled into the complete-size table above. This was a
campaign-budget stop, not evidence that either engine hit its scaling wall.
The raw file retains that stop and all completed cases. Only query comments
changed during the diagnostic; the saved source fingerprints identify the code
at its start. The probe runner subsequently gained parent-side cleanup on timeout.

The batched path used three submissions at every completed size: ownership
validation, the rewrite batch, and the final no-match check. The sequential
control used 34/66/130 submissions at 32/64/128 copies. At 128 copies the median
reduction time fell from about 572 ms to 28 ms; PyZX's median was about 23 ms.
The gain at larger sizes is specific to independent motif unions. Connected
graphs with overlapping neighborhoods can admit much smaller batches.

All 221 historical rule/full-reduce regression cases passed (220 dense tensor
comparisons against PyZX and one sampled tensor comparison), recorded in
`audit_results/gadget-pivot-batch-regression.json`. New live tests cover all
neighbor groups, existing-edge toggles, phases, overlapping centers/shared
neighbors, graph isolation, batch submission counts, both benchmark motifs,
mixed-component isomorphism, the query plan and small full reductions. The
large timing probe records degree-sequence agreement; its whole-graph tensor
and isomorphism checks exceed their bounds and remain explicitly inconclusive.
Small semantic tests support the rule; degree agreement alone is not a proof
of equivalence for the large graphs.

To repeat the bounded diagnostic, choose a new output file and an **empty,
dedicated** test server (the probe refuses a nonempty server):

```powershell
$env:ZXDB_URI = 'bolt://localhost:7688'
$env:ZXDB_RESEARCH_LIVE = '1'
python -m unittest tests.test_gadget_pivot_batch
python -m evaluation.gadget_pivot_probe --output gadget-pivot-probe-new.json --budget-seconds 180
```

The probe has a 60-second hard subprocess cap per input and a total budget;
it preserves completed inputs and stops on failure or budget exhaustion.
It is separate from the resumable publication runner. For a larger study,
create a new focused plan using the optimized variant and a new results folder:

```powershell
python -m evaluation.research plan --profile focused --families copies_pivot_gadget --db-variant optimized --output gadget-pivot-batched-plan.json
python -m evaluation.research run --plan gadget-pivot-batched-plan.json --output results-gadget-pivot-batched --restart-container memgraph --budget-seconds 1800
```

Do not resume a pre-change performance campaign with the new algorithm or mix
its timings into the historical tables. The existing focused-scale results and
LaTeX table have not been changed.

## Boundary-pivot batching (October 2026)

Optimized reduction now batches both boundary-pivot forms:

- `Pivot boundary - batched` implements `pivot_boundary_rule`: it accepts a
  general-phase boundary-adjacent spider on a simple boundary wire and creates
  its phase gadget. It retains the interior spider and its existing wires,
  reconnects the boundary, and deletes only the boundary-adjacent spider.
- `Pivot boundary Pauli - batched` implements the Pauli boundary step inside
  optimized `pivot_rule`: it accepts either boundary-wire type, toggles that
  wire, propagates both Pauli phases and does not introduce a phase gadget.

Both select disjoint closed neighborhoods, including the boundary itself.
This prevents shared-center, shared-boundary-owner and shared-neighbor
conflicts. The general-phase selector first chooses the lowest-ranked eligible
partner per interior center, avoiding duplicate neighborhood claims for
alternatives that cannot appear in the same batch. The globally minimum
candidate still guarantees progress. Neighbor matches expand bound centers;
there is no per-candidate full vertex scan. Complete winners and edge deletions
are materialized before mutation. Ownership groups carry compact
`[rank, interior, boundary-adjacent, boundary]` tuples rather than duplicating
each candidate's full neighborhood list per claimed vertex. In the local
512-copy diagnostic this reduced the median from about 102 ms to 76 ms, with
the same candidate selection. Matching order can differ from PyZX without
changing the represented map.

The ordinary `pivot_boundary_rule` remains the one-match performance control.
The single-Pauli boundary query used by ordinary `pivot_rule` received the
candidate-materialization correctness repair described above, while retaining
one application per query. The optimized paths remain explicitly opt-in.

```python
db.apply_rule_optimized("pivot_boundary_rule", graph_id, exclusive_access=True)
db.full_reduce_optimized(graph_id, exclusive_access=True)
```

Tests now cover 128 general-phase and 128 Pauli combinations of neighbor groups,
pre-existing edges, center phases and (for Pauli) both boundary-wire types;
overlapping matches; invalid candidates; ordered boundaries; foreign graphs;
small circuit full reductions; and isomorphism of mixed independent copies.
The final 221-case regression passed (220 tensor comparisons with PyZX, one
sampled tensor comparison). The broader suite passed 77 tests with one optional
test skipped. After compacting the ownership representation, all 19 boundary/
gadget tests and the 221-case regression passed again. Logs and results use
the `audit_results/boundary-pivot-*` prefix.

The final bounded probe completed all 15 inputs (three distinct motif mixtures
at each of five sizes), with one excluded warmup and two measured repetitions
per input. `audit_results/boundary-pivot-batch-probe.json` contains the raw
timings and source fingerprints. It uses the same geometric-mean paired
estimator and reduction-only timing boundary as the gadget-pivot diagnostic.
No timings from earlier implementations are pooled into these figures.

| Copies | Inputs | Measured pairs | Geometric mean PyZX / ZXdb |
|---:|---:|---:|---:|
| 32 | 3 | 6 | 0.24 |
| 128 | 3 | 6 | 0.43 |
| 512 | 3 | 6 | 0.62 |
| 1024 | 3 | 6 | 1.06 |
| 2048 | 3 | 6 | 1.15 |

ZXdb used three submissions at every size. At 32 copies it was 9.22x faster
than the sequential database control, which used 34 submissions. PyZX remained
faster at small sizes; the observed large-size advantage is modest, unlike the
gadget-pivot result. These small pilot samples do not establish statistical
significance or a general advantage on connected diagrams. Large-run output
degree sequences agreed with PyZX; whole-graph tensors/isomorphism exceeded
the bounded oracle, while small motifs and mixed-copy graphs have the stronger
semantic/structural checks above. The final probe completed without a server
failure, but the earlier failures below remain relevant.

Stability remains a limitation. An initial long-lived-server regression passed
182 cases before Memgraph 3.6.2 exited 139 on `wide_q6_d24_s1075`, without an
OOM kill. The same case passed twelve times after restart. A later scaling
probe suffered another exit 139 at size 2048, seed 1; the parent then reached
its total budget and could not clean up the dead server. These failures are
preserved in `boundary-pivot-batch-regression-initial.json` and
`boundary-pivot-batch-probe-initial.json`. The final implementation subsequently
retains the interior spider to reduce mutation work, but this is not established
as a native-crash fix. Successful fresh-run checks do not certify long-run
stability. Keep campaign recovery enabled and retain failed attempts.

The bounded diagnostic runner now supports either pivot rule and preserves
worker diagnostics if cleanup also fails:

```powershell
$env:ZXDB_URI = 'bolt://localhost:7688' # empty dedicated server
$env:ZXDB_RESEARCH_LIVE = '1'
python -m unittest tests.test_boundary_pivot_batch
python -m evaluation.gadget_pivot_probe --rule pivot_boundary --control-max-size 32 --sizes 32 128 512 1024 2048 --output boundary-pivot-probe-new.json --budget-seconds 180
```

`--control-max-size` bounds the slow sequential comparison; PyZX and optimized
ZXdb are measured at every requested size. The hard subprocess limit is 60
seconds per input, subject to the remaining campaign budget. For a publication
campaign, use a new focused plan with `--families copies_pivot_boundary
--db-variant optimized` and a new results directory. Existing focused-scale
observations, figures and the open LaTeX table are unchanged.

## Loading and the observed crash

A bounded comparison initially encountered a Memgraph 3.6.2 exit 139, with
`OOMKilled=false`, while assigning boundary labels in the research loader.
The client traceback identifies where it surfaced, not the native root cause.
The loader now creates boundary labels and ordering with the vertices instead
of subsequently matching each boundary and changing its labels. The constructor
also creates the existing vertex-property ID index used by the historical
importer, avoiding repeated endpoint scans during loading. Both control and
optimized campaigns use this loader/setup.
`EXPLAIN` showed that merely creating the index was insufficient: the planner
could still choose `graph_id` for each endpoint. The loader now explicitly hints
the vertex-ID index; a live plan regression checks both endpoint lookups.

The regression suite includes 80 repeated load/read comparisons of the affected
640-vertex, 256-boundary fixture. That check and a subsequent 221-case worklist
regression passed, but the same long-lived server later exited 139 again during
the next regression. The loader change therefore does **not** establish a fix
for the server instability. Fresh-run semantic checks and long-running stability
must be reported separately. Keep campaign recovery enabled.
Another later run stopped at `t_heavy_q3_s1064` during optimized reduction, after
171 passing cases (`optimization-index-batched.json`). All three observed server
failures had exit code 139 and were not reported as OOM kills. The native cause
and whether any optimization contributes remain unresolved; do not interpret
fresh-run passes as a stability certification for the massive campaign.
The final clean-server run with the hinted loader passed 221/221 cases
(`optimization-final-clean.json`). Ownership/Hopf/liveness checks passed 16/16;
the research/optimization suite passed 44 tests with one optional recovery test
skipped. The repeated-load and explicit index-plan checks also passed.
A later connection check after that final successful run was refused. Its exit
status was not captured before removing the disposable container. This further
limits what the successful test run establishes about server health.
The original failed diagnostic remains in `audit_results/optimization-probe.log`.

## Usage and comparison

```python
db.full_reduce(graph_id)  # control
db.full_reduce_optimized(graph_id, exclusive_access=True)
db.full_reduce_optimized(graph_id, exclusive_access=True, incremental=True)

db.apply_rule_optimized("local_complementation_rule", graph_id,
                        exclusive_access=True)
```

`exclusive_access=True` is a caller promise that no other writer changes the graph
during the call. It does not acquire a database lock. All statements remain
autocommit; no long analytical transaction has been introduced.

Research plans accept `--db-variant control`, `optimized`, or `worklist`:

```powershell
python -m evaluation.research plan --profile study --db-variant optimized --output study-optimized-plan.json
python -m evaluation.research run --plan study-optimized-plan.json --output results-study-optimized --restart-container memgraph --budget-seconds 3600

python -m evaluation.research plan --profile massive --db-variant optimized --output massive-optimized-plan.json
python -m evaluation.research run --plan massive-optimized-plan.json --output results-massive-optimized --restart-container memgraph --budget-seconds 28800
```

Use separate plans/directories for comparisons. Existing plans default to
`control`; they have not been silently changed. The variant is saved in the plan
and each database timing record. `worklist` adds frontier setup/cleanup costs:
fewer full scans need not mean lower latency. Start with `optimized` and treat
worklists as a separately measured experiment.

To repeat bounded diagnostics and the full adversarial checks on an **empty,
dedicated** test server:

```powershell
$env:ZXDB_URI = 'bolt://localhost:7688'
$env:ZXDB_RESEARCH_LIVE = '1'
python -m unittest tests.test_reduction_optimizations.ReductionOptimizationTests
python -m evaluation.optimization_regression --variant full_reduce_optimized --exclusive-access --output optimized-regression.json
python -m evaluation.optimization_regression --variant full_reduce_optimized --exclusive-access --incremental --output worklist-regression.json
python -m evaluation.optimization_probe --output optimization-probe.json
```

The small probe uses three repetitions after one warmup and alternates engine
order. It reports reductions and submissions, excluding setup/read/oracle costs.
Its pattern inputs have one seed and its realistic case is only one circuit.
Use the larger research suite, resource limits and correctness evidence before
making PyZX-relative or population-level performance claims.

## Local diagnostic results

The completed post-loader probe on the disposable Memgraph 3.6.2 server gave the
following medians (seconds; three measured repetitions). The realistic circuit
has four qubits, 40 generated gates and seed 0.

| Input | Control | Optimized batches | Worklists | Calls: control / optimized / worklist |
|---|---:|---:|---:|---:|
| 128 pivot motifs | 0.6781 | 0.0266 | 0.0356 | 131 / 4 / 7 |
| 128 local-complementation motifs | 0.6250 | 0.0156 | 0.0187 | 130 / 3 / 6 |
| Realistic full reduction | 0.4182 | 0.3208 | 0.3475 | 115 / 78 / 104 |

Batching alone was faster than worklists on these inputs. These are comparisons
between database implementations, **not speedups over PyZX**. Earlier partial
timings preceded the loader/index change and are not pooled with this completed
probe. Raw trials, output metrics and bounded semantic evidence are saved in
`audit_results/optimization-probe-after-loader.json`.
