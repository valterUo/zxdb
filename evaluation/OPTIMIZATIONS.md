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
