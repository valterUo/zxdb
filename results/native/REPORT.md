# ZXdb evaluation

Run state: finished. Profile: full.

## Correctness

{'verified': 218, 'inconclusive': 3, 'failed': 2}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits are inconclusive, never passes.

## Performance

27/30 performance cells completed without failures. 10 rewrite wins; 10 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Tensor verified |
|---|---|---:|---:|---|
| random_clifford n=32 seed=0 | native | 2.454x | 2.043x | False |
| random_clifford n=32 seed=1 | native | 2.272x | 1.971x | False |
| random_clifford n=32 seed=2 | native | 2.449x | 2.039x | False |
| random_clifford n=64 seed=0 | native | 6.484x | 5.821x | False |
| random_clifford n=64 seed=1 | native | — | — | False |
| random_clifford n=64 seed=2 | native | 4.916x | 4.356x | False |
| random_clifford_t n=32 seed=0 | native | — | — | False |
| random_clifford_t n=32 seed=1 | native | 0.522x | 0.467x | False |
| random_clifford_t n=32 seed=2 | native | 0.518x | 0.464x | False |
| random_clifford_t n=64 seed=0 | native | 1.352x | 1.243x | False |
| random_clifford_t n=64 seed=1 | native | 0.957x | 0.893x | False |
| random_clifford_t n=64 seed=2 | native | 0.981x | 0.929x | False |
| random_t_heavy n=32 seed=0 | native | 0.174x | 0.160x | False |
| random_t_heavy n=32 seed=1 | native | 0.249x | 0.219x | False |
| random_t_heavy n=32 seed=2 | native | 0.335x | 0.299x | False |
| random_t_heavy n=64 seed=0 | native | 0.281x | 0.253x | False |
| random_t_heavy n=64 seed=1 | native | — | — | False |
| random_t_heavy n=64 seed=2 | native | 0.302x | 0.278x | False |
| layered_local n=32 seed=0 | native | 0.213x | 0.187x | False |
| layered_local n=32 seed=1 | native | 0.199x | 0.176x | False |
| layered_local n=32 seed=2 | native | 0.245x | 0.208x | False |
| layered_local n=64 seed=0 | native | 0.163x | 0.153x | False |
| layered_local n=64 seed=1 | native | 0.176x | 0.164x | False |
| layered_local n=64 seed=2 | native | 0.178x | 0.166x | False |
| layered_global n=32 seed=0 | native | 1.296x | 1.097x | False |
| layered_global n=32 seed=1 | native | 0.219x | 0.192x | False |
| layered_global n=32 seed=2 | native | 1.765x | 1.512x | False |
| layered_global n=64 seed=0 | native | 0.519x | 0.460x | False |
| layered_global n=64 seed=1 | native | 4.575x | 4.036x | False |
| layered_global n=64 seed=2 | native | 3.056x | 2.739x | False |

## Failures and unresolved evidence

- correctness / full_reduce / isolated_zero_spider / native: {'db_check': {'status': 'failed', 'reason': 'Proven zero map versus nonzero tensor'}, 'pyzx_check': {'status': 'verified', 'reason': 'Explicit zero factors on both graphs'}}
- correctness / full_reduce / explicit_zero_scalar / native: {'db_check': {'status': 'failed', 'reason': 'Proven zero map versus nonzero tensor'}, 'pyzx_check': {'status': 'verified', 'reason': 'Explicit zero factors on both graphs'}}
- benchmark / full_reduce / 3172bf9db795b5ff / native: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7688)) (ResolvedIPv4Address(('127.0.0.1', 7688)))
- benchmark / full_reduce / dfc4c5f4f324fe54 / native: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7688)) (ResolvedIPv4Address(('127.0.0.1', 7688)))
- benchmark / full_reduce / b5eaaa1cff0c33ed / native: TimeoutError: Trial exceeded 120s; database interrupted

## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access and no incremental worklist. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior pivot, local complementation and gadget pivot to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking, legacy JSON import/export, and standalone supplementarity preconditions remain audit concerns. See ../../evaluation/audit-history.md.
