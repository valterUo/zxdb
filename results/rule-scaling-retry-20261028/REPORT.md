# Larger-rule retries - 2026-10-28

Status: {'complete': 3, 'error': 1}; 4/6 retry cases finished.

Fresh retries of the six failed or interrupted configurations in [the previous extension](../rule-scaling-20261028/REPORT.md). Original failures and timings remain untouched. No repetition is pooled across attempts.

Per-engine database I/O/rewrite limit: 1800 seconds. Per-case process limit: 10800 seconds. No shared campaign cutoff and no projected-runtime early stops. One warmup and three measured repetitions per engine. The dedicated database is restarted before each retry; its memory limit is unchanged. Queries, rewrite implementations, workload generators, and the native module are unchanged.

The server query.timeout setting was raised from 600 to 1800 seconds for the long retries. server-timeout-change.json records the change time; the first two cases finished without approaching the original limit. The original setting is restored after the campaign. For reproduction, set this dedicated server setting to 1800 before the command below and restore its prior value afterward.

Seconds below are medians of complete measured triplets. Speedup is PyZX/database; above one favors the database. Missing or failed triplets have no median or speedup. Total includes graph import/export (PyZX: copying), but excludes startup and verification. Timing completion is distinct from semantic verification; large tensor checks can remain inconclusive.

| Workload | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher rewrite speedup | Hybrid rewrite speedup | Cypher total speedup | Hybrid total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| copies_lcomp | 4096 | 11.4336 | 22.0537 | 0.13091 | 0.518443 | 87.3395 | 0.477304 | 5.34902 |
| pivot_boundary | 512 | 0.48949 | 0.408806 | 0.428827 | 1.19736 | 1.14146 | 0.0604861 | 0.0607709 |
| bialgebra | 512 | -- | -- | -- | -- | -- | -- | -- |
| copies_bialgebra | 2048 | 0.980734 | 210.261 | 208.012 | 0.00466435 | 0.00471479 | 0.00484622 | 0.00490937 |

## Retry outcomes

- copies_lcomp, n=4096: complete (157.1 seconds).
- pivot_boundary, n=512: complete (86.8 seconds).
- bialgebra, n=512: error (264.0 seconds).
- copies_bialgebra, n=2048: complete (1716.4 seconds).

## Errors, including warmups

- 002-bialgebra-512, hybrid, repetition 0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.25GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- 002-bialgebra-512, cypher, repetition 0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.31GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- 002-bialgebra-512: container exit code 0, OOMKilled=False.

## Reproduction

```sh
python -m evaluation.extend_rule_scaling --retry-from results/rule-scaling-20261028 --output results/new-retry --budget 0 --case-timeout 10800 --engine-timeout 1800
python results/rule-scaling-retry-20261028/summarize.py
```

Each case retains its source archive, environment, input/output graphs, paired observations and immediate per-engine observations. The immediate stream preserves evidence if a later engine crashes; the summary uses only the paired observations.
