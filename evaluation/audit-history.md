# Audit status

The original detailed audit and raw logs are retained in the backup repository
`../zxdb-backup-20261026`. This file records which findings the consolidated
evaluation addresses. Current measurements and validation are indexed in
[`results/README.md`](../results/README.md); older reports describe their own revisions.

| Finding | Current status and evidence |
|---|---|
| Rewrites ignored diagram ownership | Fixed previously. Graph-isolation regressions cover rule fixtures, missing IDs, crossing edges and native-wrapper refusal. |
| Hopf cancelled the wrong wire types | Fixed previously. Normalization regressions cover both colors, multiplicities and fusion counterexamples. |
| JSON import silently does nothing by default or clears all data | Still present in the legacy API. The evaluator uses the scoped loader in `evaluation/core.py` and never calls whole-database deletion. |
| Zero maps become nonzero | The evaluation loader represents explicit zero scalars as isolated Z(pi) components, which cleanup now preserves. Both cases pass current regressions. General scalar tracking remains incomplete; nonzero global factors are intentionally ignored. |
| Correctness checker could hide scalar errors | Replaced for the canonical evaluation. The degree-zero spider tensor now adds both terms; scalar factors are included. Each output is compared against the original, and unsupported or oversized tensors are inconclusive. Independent PyZX tensor tests cover both colors and zero arity. Ambiguous near-zero/nonfinite contractions are inconclusive, and zero-map failures require exact zero factors. |
| Standalone supplementarity lacks graph-like preconditions | Fixed in the 2026-10-27 audit: both Cypher variants reject simple incident wires, non-Z neighbors, self-loops and duplicate wires. See the new rule audit tests and report. |
| Legacy export changes multigraphs | Still present in the visualization-oriented API. The evaluation reader preserves parallel edges, self-loops and ordered boundaries. |
| Hadamard-gate conversion references a missing query | Legacy API limitation, distinct from the evaluated Hadamard cancellation rule. |
| Mutation is not atomic or comprehensively bounded | Analytical mode is required by current queries. Memgraph 3.6.2 crashed during this evaluation. Optional dedicated-container recovery records the failed trial and continues; it does not make the implementation reliable or atomic. |
| Fragmented reproducibility and query coverage | One runner now records fixed workloads, hashes, versions, warmups, repeated timings, failures and evidence levels for all eleven rule families and full_reduce. Auxiliary behaviors remain covered by regressions. |

The native variant now uses six graph-scoped C++ procedures and participates in
the same evaluation pipeline. Native regressions cover rule fixtures, boundary
cases, zero factors and graph isolation. The older zero-argument gadget procedure
remains a legacy comparison path. Incremental worklist experiments were removed.
The results index distinguishes verified outputs from large timing-only evidence.

Historical campaign summaries are retained in `results/historical-summary.json`.
Their pass totals and timings must not be mixed with the current run: versions,
workloads and verification machinery differed. A database win on independent
synthetic patterns is not evidence of a win on full circuit reduction.

The [2026-10-27 rule audit](../results/rule-audit-20261027/REPORT.md) records
additional fixes for gadget fusion, self-loop normalization, identity progress,
legacy bialgebra, and the retained Python gadget module. Its regression totals
are separate from the historical timing campaigns above.
