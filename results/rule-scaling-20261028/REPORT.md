# Larger standalone-rule timings - 2026-10-28

Campaign status: {'complete': 23, 'error': 5, 'case_timeout': 1, 'budget_not_run': 40, 'skipped_after_failure': 13}; 82/82 jobs accounted for.

Three measured repetitions after one warmup for each completed case. PyZX, Cypher, and hybrid share the same input, with deterministically shuffled engine order. All timings below are medians. Old campaign measurements are not pooled.

Queries, rewrite implementations, native module and graph generators are unchanged. Only benchmark sizes increase. Dense fan sizes are capped at 1024 because outputs can have quadratically many edges. Other workloads are planned up to size 16384. A one-hour campaign budget and 1200-second case deadline bound resource use. The final case deadline is shortened to the remaining campaign budget; a case_timeout is not an engine-specific 1200-second runtime lower bound. A failure stops larger sizes of that workload family, regardless of which engine failed. No stopping criterion depends on a database win. During this run, resource-based early stops were also applied when warmup duration projected beyond the case budget; resource-stop.json files record those decisions. These manual early stops are not automatically reproduced by the command below, so the reached subset may differ.

Total includes in-memory graph import/export, or PyZX graph copying, but excludes startup, graph snapshots and verification. A timed completion is not a semantic certificate. Large tensor checks remain inconclusive; see correctness-summary.json and per-trial checks. Missing, failed or incomplete measurements are not reported as wins.

| Workload | Size n | Vertices | PyZX rewrite (s) | Cypher rewrite (s) | Hybrid rewrite (s) | Cypher rewrite speedup | Hybrid rewrite speedup | Cypher total speedup | Hybrid total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copies_copy | 2048 | 8192 | 0.25477 | 19.563 | 20.443 | 0.013024 | 0.012462 | 0.014558 | 0.013906 |
| copies_gadget_fusion | 2048 | 16384 | 0.28083 | 0.13656 | 0.13365 | 2.0565 | 2.1012 | 0.1744 | 0.17901 |
| copies_identity | 2048 | 12288 | 7.7555 | 0.080203 | 0.088032 | 96.698 | 88.099 | 5.2063 | 5.8724 |
| copies_identity | 4096 | 24576 | 29.542 | 0.13972 | 0.16591 | 211.44 | 178.06 | 11.522 | 10.925 |
| copies_lcomp | 2048 | 10240 | 3.2299 | 3.7868 | 0.075819 | 0.85295 | 42.601 | 0.6871 | 2.7048 |
| copies_pivot | 2048 | 20480 | 5.7009 | 6.688 | 0.20269 | 0.85241 | 28.127 | 0.621 | 2.4378 |
| copies_pivot_boundary | 2048 | 22528 | 0.32673 | 0.34157 | 0.21964 | 0.95654 | 1.4875 | 0.11649 | 0.11814 |
| copies_pivot_gadget | 2048 | 20480 | 4.0704 | 6.6602 | 0.22907 | 0.61115 | 17.77 | 0.4472 | 1.3537 |
| copies_spider_fusion | 2048 | 16384 | 12.304 | 0.23061 | 0.24075 | 53.355 | 51.108 | 5.3537 | 4.9338 |
| copies_spider_fusion | 4096 | 32768 | 39.136 | 0.33264 | 0.33979 | 117.65 | 115.18 | 10.5 | 10.179 |
| copy | 512 | 514 | 0.0085085 | 0.012743 | 0.018525 | 0.6677 | 0.45929 | 0.097712 | 0.08001 |
| gadget_fusion | 512 | 1028 | 0.0025475 | 0.019899 | 0.020267 | 0.12802 | 0.1257 | 0.075023 | 0.079999 |
| hadamard_cancel | 512 | 1536 | 0.11628 | 4.9764 | 5.2363 | 0.023366 | 0.022206 | 0.023494 | 0.022333 |
| identity | 512 | 1536 | 0.11307 | 0.02 | 0.017292 | 5.6535 | 6.5389 | 0.73013 | 0.78965 |
| identity | 1024 | 3072 | 0.47208 | 0.028224 | 0.025221 | 16.726 | 18.717 | 1.5706 | 1.6567 |
| identity_chain | 2048 | 2050 | 0.24467 | 0.082831 | 0.085314 | 2.9538 | 2.8678 | 1.3115 | 1.2703 |
| lcomp | 512 | 1025 | 0.16844 | 0.17946 | 0.19392 | 0.93861 | 0.8686 | 0.038007 | 0.038212 |
| lcomp | 1024 | 2049 | 0.63183 | 0.64376 | 0.71843 | 0.98146 | 0.87945 | 0.037393 | 0.036904 |
| pivot | 512 | 2050 | 0.92886 | 0.43993 | 0.83841 | 2.1114 | 1.1079 | 0.099387 | 0.09908 |
| pivot_gadget | 512 | 2050 | 13.685 | 0.37574 | 0.45654 | 36.42 | 29.975 | 1.5003 | 1.4556 |
| spider_fusion | 512 | 2048 | 0.2016 | 0.03428 | 0.03394 | 5.8811 | 5.94 | 0.76429 | 0.71557 |
| spider_fusion | 1024 | 4096 | 0.64294 | 0.052806 | 0.05589 | 12.175 | 11.504 | 1.4124 | 1.4494 |
| supplementarity | 512 | 514 | 0.0069567 | 0.28687 | 0.22183 | 0.024251 | 0.03136 | 0.029282 | 0.035499 |

Speedup = PyZX/database; above 1 favors the database. `copies_*` measures disjoint size-2 motifs. Other family sizes retain their existing generator meaning; fan size is not a qubit count.

## Uncompleted configurations

- pivot_boundary, n=512: error
- bialgebra, n=512: error
- copies_bialgebra, n=2048: error
- copies_hadamard_cancel, n=2048: error
- copies_supplementarity, n=2048: error
- copies_lcomp, n=4096: case_timeout
- pivot, n=1024: budget_not_run
- copies_pivot, n=4096: budget_not_run
- pivot_gadget, n=1024: budget_not_run
- copies_pivot_gadget, n=4096: budget_not_run
- pivot_boundary, n=1024: skipped_after_failure
- copies_pivot_boundary, n=4096: budget_not_run
- gadget_fusion, n=1024: budget_not_run
- copies_gadget_fusion, n=4096: budget_not_run
- bialgebra, n=1024: skipped_after_failure
- copies_bialgebra, n=4096: skipped_after_failure
- hadamard_cancel, n=1024: budget_not_run
- copies_hadamard_cancel, n=4096: skipped_after_failure
- supplementarity, n=1024: budget_not_run
- copies_supplementarity, n=4096: skipped_after_failure
- copy, n=1024: budget_not_run
- copies_copy, n=4096: budget_not_run
- identity_chain, n=4096: budget_not_run
- spider_fusion, n=4096: budget_not_run
- copies_spider_fusion, n=8192: budget_not_run
- identity, n=4096: budget_not_run
- copies_identity, n=8192: budget_not_run
- copies_lcomp, n=8192: skipped_after_failure
- copies_pivot, n=8192: budget_not_run
- copies_pivot_gadget, n=8192: budget_not_run
- copies_pivot_boundary, n=8192: budget_not_run
- gadget_fusion, n=4096: budget_not_run
- copies_gadget_fusion, n=8192: budget_not_run
- copies_bialgebra, n=8192: skipped_after_failure
- hadamard_cancel, n=4096: budget_not_run
- copies_hadamard_cancel, n=8192: skipped_after_failure
- supplementarity, n=4096: budget_not_run
- copies_supplementarity, n=8192: skipped_after_failure
- copy, n=4096: budget_not_run
- copies_copy, n=8192: budget_not_run
- identity_chain, n=8192: budget_not_run
- spider_fusion, n=16384: budget_not_run
- copies_spider_fusion, n=16384: budget_not_run
- identity, n=16384: budget_not_run
- copies_identity, n=16384: budget_not_run
- copies_lcomp, n=16384: skipped_after_failure
- copies_pivot, n=16384: budget_not_run
- copies_pivot_gadget, n=16384: budget_not_run
- copies_pivot_boundary, n=16384: budget_not_run
- gadget_fusion, n=16384: budget_not_run
- copies_gadget_fusion, n=16384: budget_not_run
- copies_bialgebra, n=16384: skipped_after_failure
- hadamard_cancel, n=16384: budget_not_run
- copies_hadamard_cancel, n=16384: skipped_after_failure
- supplementarity, n=16384: budget_not_run
- copies_supplementarity, n=16384: skipped_after_failure
- copy, n=16384: budget_not_run
- copies_copy, n=16384: budget_not_run
- identity_chain, n=16384: budget_not_run

## Recorded errors (including warmups)

- bialgebra, n=512, hybrid, repetition 0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.25GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- bialgebra, n=512, cypher, repetition 0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.31GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- copies_supplementarity, n=2048, cypher, repetition 0: TimeoutError: Database rewrite/I/O exceeded 300s; interruption requested
- 010-pivot_boundary-512: container exit code 139; Docker OOMKilled=False. See the preserved container log.
- 015-copies_bialgebra-2048: early resource stop. Warmup engine rewrite times alone total 423.1 seconds; projected four repetitions exceed 1200-second case budget. Stop early to preserve remaining campaign budget.
- 017-copies_hadamard_cancel-2048: early resource stop. Warmup still incomplete after more than 480 seconds; projected warmup plus three measured repetitions exceeds 1200-second case budget. Stop to preserve remaining campaign budget.

## Figures and validation

[Updated four-page TikZ figure preview](figures/main.pdf). [Combined plot inputs](combined-timings.csv) include each row's source campaign. Each configuration is taken from exactly one campaign; repetitions are not pooled. The figures include earlier smaller sizes and completed larger sizes. Missing larger cases remain listed above.

[Summary validation](summary-validation.json) checks every reported median and ratio against the raw trials. [Source integrity](integrity.json) and [comparison to the earlier source](baseline-source-comparison.json) record unchanged implementations.

## Reproduction

```sh
python -m evaluation.extend_rule_scaling --output results/new-rule-scaling --budget 3600
python results/rule-scaling-20261028/summarize.py
python results/rule-scaling-20261028/make_figures.py
```

The first command requires the existing dedicated native-enabled database on port 7691. The second rebuilds this saved campaign summary. Each case directory retains its manifest, source archive, environment, graph snapshots, trial observations and run log. plan.json records the full grid and progress.json records resource-limited omissions.
