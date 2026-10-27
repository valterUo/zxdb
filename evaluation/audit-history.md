# Audit status

The original detailed audit and raw logs are retained in the backup repository
`../zxdb-backup-20261026`. This file records which findings the consolidated
evaluation addresses. Current measurements are in `results/full/REPORT.md`.

| Finding | Current status and evidence |
|---|---|
| Rewrites ignored diagram ownership | Fixed previously. Graph-isolation regressions cover rule fixtures, missing IDs, crossing edges and native-wrapper refusal. |
| Hopf cancelled the wrong wire types | Fixed previously. Normalization regressions cover both colors, multiplicities and fusion counterexamples. |
| JSON import silently does nothing by default or clears all data | Still present in the legacy API. The evaluator uses the scoped loader in `evaluation/core.py` and never calls whole-database deletion. |
| Zero maps become nonzero | Still present in database scalar handling. Explicit zero scalars and isolated Z(pi) components are now mandatory full-reduce cases; failures are reported. Nonzero global factors are intentionally ignored. |
| Correctness checker could hide scalar errors | Replaced for the canonical evaluation. The degree-zero spider tensor now adds both terms; scalar factors are included. Each output is compared against the original, and unsupported or oversized tensors are inconclusive. Independent PyZX tensor tests cover both colors and zero arity. Ambiguous near-zero/nonfinite contractions are inconclusive, and zero-map failures require exact zero factors. |
| Standalone supplementarity lacks graph-like preconditions | Remains an implementation limitation. Passing the implemented fixtures does not establish soundness for arbitrary non-graph-like inputs. |
| Legacy export changes multigraphs | Still present in the visualization-oriented API. The evaluation reader preserves parallel edges, self-loops and ordered boundaries. |
| Hadamard-gate conversion references a missing query | Legacy API limitation, distinct from the evaluated Hadamard cancellation rule. |
| Mutation is not atomic or comprehensively bounded | Analytical mode is required by current queries. Memgraph 3.6.2 crashed during this evaluation. Optional dedicated-container recovery records the failed trial and continues; it does not make the implementation reliable or atomic. |
| Fragmented reproducibility and query coverage | One runner now records fixed workloads, hashes, versions, warmups, repeated timings, failures and evidence levels for all eleven rule families and full_reduce. Auxiliary behaviors remain covered by regressions. |

The optional native module has not received the same semantic/performance audit
as the Cypher implementation. Its Python wrapper rejects mixed-graph databases,
but the native procedure itself remains database-wide. The canonical benchmark
compares control and optimized Cypher paths; incremental worklists are tested as
regressions but are not a third performance variant.

Historical campaign summaries are retained in `results/historical-summary.json`.
Their pass totals and timings must not be mixed with the current run: versions,
workloads and verification machinery differed. A database win on independent
synthetic patterns is not evidence of a win on full circuit reduction.
