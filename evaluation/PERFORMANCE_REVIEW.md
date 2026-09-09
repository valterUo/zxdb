# Performance review of the completed study

Reviewed 2026-09-08, using `results-study` and the current source tree. This is
an analysis and optimization proposal; no production optimization was enabled.

## Findings

The campaign records 268 completed blocks and 1,340 measured paired trials.
Completion is operational: five measured trials report semantic failure, and six
worker `ServiceUnavailable` records remain. The summary contains 14 verified,
86 inconclusive and eight failed/incomplete cells. Restarting and completing a
block does not establish correctness or erase its failures.

The reported paired speedups below are **PyZX-wrapper time / ZXdb time**, not
necessarily native PyZX API performance. These large cells are semantically
inconclusive under the bounded tensor oracle. Their intervals cover timing
variability on one constructed input, not workload-population uncertainty.

| Family / size | Reported speedup | 95% interval |
|---|---:|---:|
| Pivot gadget / 128 neighbors | 6.93 | 5.11–7.78 |
| Pivot / 128 neighbors | 3.60 | 1.97–3.96 |
| Bialgebra / 128 neighbors | 2.97 | 2.65–3.18 |
| Copies of identity motif / 128 copies | 3.61 | 3.50–3.68 |
| Copies of spider motif / 128 copies | 2.59 | 2.38–2.68 |

These results suggest a useful operating region, but the reference-timing issue
below must be resolved before making claims about outperforming PyZX.

## Correct the measurement before optimizing

### PyZX adapter overhead and stopping contracts

`research_core.reference` snapshots the entire graph, runs a simplifier, compares
the snapshot, and repeats until unchanged. The installed PyZX 0.10.3 single- and
double-vertex simplifiers already contain their own fixed-point loops. The native
`full_reduce` routine also owns its reduction schedule. Repeating those complete
routines adds redundant scans and fingerprinting; in the full-reduce comparison
it can also change the algorithmic stopping contract compared with ZXdb's one
invocation of the matching reduction schedule.

However, PyZX's `RewriteSimpGraph` adapters for pivot gadget, pivot boundary and
gadget fusion execute a matching batch rather than a general fixed-point loop.
Do not blindly replace every reference by one call: compare equivalent contracts,
and verify progress/stability outside the measured interval where possible.

A local diagnostic at size 128 (three measured repetitions after one warmup)
gave these medians. Both paths ran locally in Python; this is not a replacement
for the paired study and was not run simultaneously with its recorded DB timings.

| Workload | Native call (s) | Current adapter (s) |
|---|---:|---:|
| copies_identity | 0.03009 | 0.03263 |
| copies_spider_fusion | 0.04230 | 0.05091 |
| pivot | 0.03751 | 0.17123 |
| pivot_gadget | 0.01524 | 0.30326 |
| bialgebra | 2.74195 | 5.45387 |

On these particular inputs, a second native call left the exact payload unchanged.
For pivot gadget that does not prove one call suffices on arbitrary inputs. The
large discrepancy does show why the present apparent wins need qualification.
Use one native `pyzx.full_reduce` invocation as the primary full-reduce baseline,
and explicitly document any stronger fixed-point variant as a separate algorithm.

### False zero-map classification

All five measured semantic failures are `gadget_fusion`, size 128. Both output
graphs are exactly isomorphic. The saved PyZX scalar is `power2=-127`, giving the
nonzero factor approximately `7.666e-20`. Its maximum tensor entry is `5.421e-20`;
the DB tensor's maximum entry is about `0.7071` because the DB does not track that
scalar. The absolute `1e-12` zero threshold misclassifies the PyZX tensor as zero.
Reconstructing both saved outputs and removing their nonzero scalar factors gave
identical tensors, with maximum absolute difference zero.

The projective oracle should separate exact/symbolic zero scalar detection from
normalization of nonzero global scale, before floating-point contraction. Unknown
or near-cancelling scalar expressions should remain inconclusive. Simply disabling
zero checks would reintroduce earlier zero-map bugs. Preserve the original results
and record any corrected revalidation separately.

## Main performance bottlenecks

### Many small statements and repeated global matching

| Rule on 128 independent motif copies | Cypher submissions | DB median (s) | PyZX-wrapper median (s) |
|---|---:|---:|---:|
| Identity | 5 | 0.00895 | 0.03181 |
| Spider fusion | 7 | 0.01804 | 0.04767 |
| Local complementation | 130 | 0.20037 | 0.01878 |
| Pivot | 131 | 0.45381 | 0.03743 |
| Pivot gadget | 130 | 0.47930 | 0.02409 |
| Pivot boundary | 130 | 0.90480 | 0.02132 |
| Bialgebra | 258 | 1.31872 | 0.03634 |

The successful batched rules keep the submission count roughly constant as pattern
count grows. The others repeatedly choose a small number of matches, rewrite,
and search the graph again. This incurs both statement overhead and increasingly
expensive scans; eliminating transport alone will not cure repeated matching.

Read-only probes on the current Windows-to-Docker connection, 50 observations
after five warmups, measured `RETURN 1` at 0.824 ms median and an empty ownership
guard at 0.932 ms. These include driver/protocol/server dispatch costs, not just
network latency. At 100–170 submissions, a rough fixed-cost reference is 82–140 ms.
It is not a measured decomposition of each study trial, but explains why very
small full reductions struggle against an entirely in-process library.

### Current full-reduce inputs are modest

Random circuits vary width up to 16 but use fixed generator depths 20 and 80.
Increasing qubits at fixed depth does not proportionally increase rewrite work.
For example, a 16-qubit, two-layer structured case has 230 vertices; the four-layer
local case has 412. This is useful coverage, but not evidence about full reduction
on tens of thousands or millions of vertices.

## Prioritized changes

1. **Correct the reference contract and projective oracle.** Add regressions for
   redundant fixed-point scans, tiny nonzero scalars and true zero maps. Report
   native and stronger fixed-point variants separately where necessary. Preserve
   all original campaign observations.

2. **Move the reduction loop into Memgraph.** The existing C++ module is the best
   starting point, but currently implements only pivot-gadget fixpoint and takes
   no graph_id argument. A graph-scoped native `full_reduce(graph_id)` can avoid
   client calls between every match/rewrite and maintain candidate queues in
   process. Audit its lifetime handling, multiedge assumptions and graph ownership
   first; this is not a recommendation to enable the current module unchanged.
   The native path should run the same schedule and report the same quality metrics.

3. **Batch independent rewrites without an all-pairs conflict computation.** Start
   with local complementation and pivot on `copies_*`, then test overlapping
   circuits. Build a greedy set of candidates with disjoint affected neighborhoods
   using visited/claimed sets. Collect matches before mutation and revalidate
   eligibility when necessary. Merely removing `LIMIT 1` is not safe. The existing
   experimental batched-pivot query has an expensive conflict-selection strategy;
   its source records earlier slowdowns. Fewer statements alone is not enough.

4. **Replace full rescans by updates around changed vertices.** Maintain a work
   queue of potentially eligible spiders and refresh only affected neighborhoods.
   For a Cypher-only variant, investigate staged candidate IDs / temporary candidate
   state, with explicit invalidation and graph_id scoping. Measure its maintenance
   cost. A native queue is likely simpler and cheaper than persisting every queue
   operation as another database mutation.

5. **Reduce provably redundant checks in the current Cypher pipeline.** Validate
   graph isolation once at a private full-reduce boundary under an explicit
   no-concurrent-writers contract; retain checks on standalone public rules. Reuse
   established color/normalization invariants only until a rule can invalidate
   them. Consider a combined termination probe instead of many separate no-match
   calls. Preserve the rule schedule; fewer checks must not mean weaker reduction.

6. **Profile actual hot queries before changing indexes.** Capture `EXPLAIN` and
   `PROFILE` on graph snapshots representative of expensive rewrites, outside the
   measured campaign. Inspect candidate cardinality, global sorting before LIMIT,
   repeated expansions and edge- versus node-index starts. Property indexes do not
   automatically accelerate per-node adjacency operations. Composite indexes or
   materialized phase classes are experiments with maintenance costs, not blanket
   recommendations. Memgraph documents operator timings in its
   [query-plan guide](https://memgraph.com/blog/how-to-read-query-execution-plans-optimize-queries).

7. **Test client/server co-location in the same Linux environment.** Run the Python
   harness inside WSL or a container near Memgraph, compare the lightweight-query
   baseline, then repeat the identical fixed cases. Run PyZX there too, with the
   same CPU budget and dependency versions. Reducing the observed 0.8 ms statement
   cost could help, but no improvement is established until measured. Extra RAM
   or sleep between calls is not supported as a solution by the current evidence.

## Next controlled study

Keep the existing graph families and seeds. Compare the current implementation,
corrected reference measurement, co-located deployment, and one candidate
optimization at a time. Extend depth as a separate predeclared axis (for example
100, 1,000 and 10,000 generator steps) at fixed widths; extend independent-pattern
counts as a separate experiment. Keep dense, sparse and overlapping negative
controls. Enforce timeout/memory bounds and report failed attempts.

The key measurements are native-PyZX-relative reduction time, per-rule submissions,
server operator time, matcher time versus mutation time, peak intermediate graph
size, output quality and semantic evidence. Multiple independent runs are needed
before projecting crossovers. There is no evidence yet that a Cypher-only
full-reduce pipeline will consistently beat PyZX on ordinary sparse circuits;
the graph-scoped native path has the strongest architectural opportunity.
