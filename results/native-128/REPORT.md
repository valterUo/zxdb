# ZXdb evaluation

Run state: partial. Profile: full.

## Correctness

{'verified': 220, 'inconclusive': 3}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. Proofs run outside the timing/watchdog region and are cached by input/output hashes.

## Performance

0/1 performance cells completed without failures. 0 rewrite wins; 0 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |
|---|---|---:|---:|---|
| random_clifford n=128 seed=0 | native | — | — | False |

## Failures and unresolved evidence

- regressions /  / unittest / : Regression subprocess exit code 1; see regressions.log.
- benchmark / full_reduce / 9f0326852fc2e119 / native: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7688)) (ResolvedIPv4Address(('127.0.0.1', 7688)))
- pipeline /  / stopped_for_color_change_fix / : Campaign stopped after a connection loss in Color change edge toggle. Results preserved; materialization fix requires a fresh campaign.

## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access and no incremental worklist. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking, legacy JSON import/export, and standalone supplementarity preconditions remain audit concerns. See ../../evaluation/audit-history.md.
