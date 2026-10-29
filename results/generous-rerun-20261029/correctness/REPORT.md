# ZXdb evaluation

Run state: finished. Profile: full.

## Correctness

{'verified': 440, 'inconclusive': 6}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. Proofs run outside the timing/watchdog region and are cached by input/output hashes.

## Performance

0/0 performance cells completed without failures. 0 rewrite wins; 0 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |
|---|---|---:|---:|---|

## Failures and unresolved evidence


## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking and legacy JSON import/export remain audit concerns. See [audit history](../../../evaluation/audit-history.md).
