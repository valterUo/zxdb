# ZXdb evaluation

Run state: finished. Profile: full.

Post-run validation: all 223 correctness fixtures are verified when the
[supplemental exact checks](supplemental-correctness.json) are included. All
[66 post-integration regressions](post-integration-regressions.log) passed.
Large-output certificates cover all 37 distinct measured input/output pairs,
representing all 45 measured trials. The last two reference-based proofs took
1,296 and 1,305 seconds, outside all benchmark timers.
See [combined semantic coverage](semantic-coverage.json) and the
[completion audit](completion-audit.json). Original observations from the timing run
below remain unchanged, including inconclusive tensor checks. These certificates
establish equivalence up to a nonzero global scalar, not complete scalar tracking.

## Correctness

{'verified': 220, 'inconclusive': 3}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. Proofs run outside the timing/watchdog region and are cached by input/output hashes.

## Performance

15/15 performance cells completed without failures. 15 rewrite wins; 15 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |
|---|---|---:|---:|---|
| random_clifford n=128 seed=0 | optimized | 8.039x | 7.643x | False |
| random_clifford n=128 seed=1 | optimized | 7.364x | 7.025x | False |
| random_clifford n=128 seed=2 | optimized | 8.493x | 8.048x | False |
| random_clifford_t n=128 seed=0 | optimized | 16.890x | 13.601x | False |
| random_clifford_t n=128 seed=1 | optimized | 9.496x | 8.159x | False |
| random_clifford_t n=128 seed=2 | optimized | 14.038x | 11.530x | False |
| random_t_heavy n=128 seed=0 | optimized | 1.622x | 1.240x | False |
| random_t_heavy n=128 seed=1 | optimized | 1.692x | 1.253x | False |
| random_t_heavy n=128 seed=2 | optimized | 1.668x | 1.279x | False |
| layered_local n=128 seed=0 | optimized | 1.712x | 1.294x | False |
| layered_local n=128 seed=1 | optimized | 1.613x | 1.263x | False |
| layered_local n=128 seed=2 | optimized | 1.583x | 1.255x | False |
| layered_global n=128 seed=0 | optimized | 63.837x | 50.525x | False |
| layered_global n=128 seed=1 | optimized | 202.404x | 161.676x | False |
| layered_global n=128 seed=2 | optimized | 72.407x | 56.514x | False |

## Duplicate seed inputs

These seed settings generate identical input graphs; they are repeated measurements, not independent workloads.

- layered_local n=128: seeds [0, 1, 2]

## Follow-up reference reduction

Counts below deduplicate exact saved outputs. This untimed structural check is separate from semantic equivalence.

- db: {'unchanged': 49}
- pyzx: {'unchanged': 12, 'inconclusive': 1}

## Failures and unresolved evidence


## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking, legacy JSON import/export, and standalone supplementarity preconditions remain audit concerns. See [audit history](../../../evaluation/audit-history.md).
