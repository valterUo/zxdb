# ZXdb evaluation results

## Current Cypher-only repeated comparison

The [completed timing campaign](cypher-optimization/neighborhood-cost-repeated/REPORT.md)
has 15/15 median rewrite wins and 15/15 median total-time wins across five
families at 128 qubits, with three measured repetitions per configuration.
All 45 measured database outputs passed unchanged structural fixed-point audits.
The 15 configurations contain 13 distinct inputs: local-layered seed labels
produce the same graph. All 25 recorded source hashes matched the working tree
after the run. The server is stock Memgraph 3.14.0 without custom rewrite procedures.

The following ranges span the three configuration medians within each family;
they are not confidence intervals. Ratios above one favor Cypher.

| Family | Rewrite speedup | Total-time speedup |
|---|---:|---:|
| Clifford | 7.36–8.49x | 7.02–8.05x |
| Clifford+T | 9.50–16.89x | 8.16–13.60x |
| T-heavy | 1.62–1.69x | 1.24–1.28x |
| Local-layered | 1.58–1.71x | 1.25–1.29x |
| Global-layered | 63.84–202.40x | 50.52–161.68x |

All 65 regression tests passed. The main run verified 220/223 correctness
fixtures; [supplemental checks](cypher-optimization/neighborhood-cost-repeated/supplemental-correctness.json)
verified the remaining three using exact symbolic zero tensors and unitary
identity certificates. Original inconclusive tensor observations are preserved.
The completed reference audit verified 19 of 37 distinct measured input/output
pairs, covering 27 of 45 measured trials: Clifford, T-heavy and local-layered.
The other 18 attempts timed out (Clifford+T and global-layered). A subsequent
[direct global-layered proof](cypher-optimization/neighborhood-cost-repeated/direct-layered_global-first-600s-certificates.json)
verified one more output in 329 seconds; further direct proofs have also passed.
The current combined counts and exact graph hashes are in the
[coverage report](cypher-optimization/neighborhood-cost-repeated/semantic-coverage.json).
All 37 distinct measured input/output pairs now have certificates, covering all
45 measured trials. The last two global-layered outputs passed with a
1,800-second bound, taking 1,296 and 1,305 seconds. Previous timeouts remain
in the evidence; they have not been rewritten as successes.
The Clifford+T representative passed using an audit-only
order-preserving gadget matcher with cached eligibility and incident edges;
120 differential cases checked its matches and mutations against unmodified PyZX.
Installed and timed PyZX remain unchanged. Per-proof reports record this reliance.
The cached matcher is now available in the canonical certificate subprocess;
[integration provenance](cypher-optimization/neighborhood-cost-repeated/post-run-oracle-integration.json)
records this post-timing change and its regression checks.
All 66 post-integration regressions passed with no skips; the
[full log](cypher-optimization/neighborhood-cost-repeated/post-integration-regressions.log)
includes the database checks and isolated-certificate counterexamples.
These counts cover measured trials, not additional warmup outputs. Matching order and
resulting graph density contribute to these speedups; smaller workloads can
still favor PyZX. Total time includes in-memory import/export, not disk I/O or startup.

## Reproduction and interpretation

Use the single [evaluation runner](../evaluate.py); the [root README](../README.md)
contains the dedicated-server command. The campaign folder retains all raw
observations, graph snapshots, source archives, versions and regression output.
The query implementation is [main_queries.json](../zxdb/main_queries.json),
coordinated by the optimized schedule in [zxdb.py](../zxdb/zxdb.py).

Correctness coverage includes 115 full-reduction fixtures and 108 individual-rule
fixtures across all eleven rule families. Regression tests additionally exercise
edge parity, overlapping matches, eligibility-marker cleanup and graph isolation.
Equivalence is checked up to a nonzero global scalar; general scalar tracking is
not implemented. Standalone supplementarity retains the preconditions documented
in the [audit history](../evaluation/audit-history.md).

The measured large-output audit is complete. See its
[reference certificates](cypher-optimization/neighborhood-cost-repeated/reference-certificates.json)
and [direct global-layered attempts](cypher-optimization/neighborhood-cost-repeated/direct-layered_global-certificates.json).
Certificates against a reduced reference rely on trusted PyZX reduction of the
same hash-checked unitary input. Direct proof attempts, timeouts and rejected
experimental proof shortcuts are preserved separately. A structural fixed point
alone is not a semantic certificate.

## Historical evidence and backup

Earlier optimization experiments are in the independent backup repository:
`../../zxdb-backup-20261026/cypher-development-20261027/`.
The [previous results index](../../zxdb-backup-20261026/cypher-development-20261027/results-index-before-consolidation.md)
retains their full interpretation. Cleanup verified 3,550 experiment files and
24 metadata files against SHA-256 backups before removing the originals;
[experiment manifest](cypher-optimization/development-backup-manifest.json) and
[metadata manifest](cypher-optimization/metadata-backup-manifest.json) record them.

Earlier native and repair runs are preserved in
`../../zxdb-backup-20261026/validation-history-20261027/`; the
[archive record](archived-runs.json) identifies 2,048 hash-verified files.
Their counts and timings describe other revisions and must not be mixed with
the current Cypher-only run.
[Historical summaries](historical-summary.json) and the
[native build record](memgraph-3.14/build.json) remain available for comparison.
