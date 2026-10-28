# ZXdb evaluation

Run state: finished. Profile: quick.

## Correctness

{'verified': 440, 'inconclusive': 6}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. Proofs run outside the timing/watchdog region and are cached by input/output hashes.

## Performance

76/76 performance cells completed without failures. 0 rewrite wins; 0 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |
|---|---|---:|---:|---|
| copies_spider_fusion n=2 seed=0 | optimized | 0.017x | 0.018x | True |
| copies_spider_fusion n=2 seed=0 | native | 0.016x | 0.017x | True |
| copies_spider_fusion n=8 seed=0 | optimized | 0.053x | 0.040x | False |
| copies_spider_fusion n=8 seed=0 | native | 0.058x | 0.043x | False |
| spider_fusion n=2 seed=0 | optimized | 0.008x | 0.010x | True |
| spider_fusion n=2 seed=0 | native | 0.007x | 0.009x | True |
| copies_identity n=2 seed=0 | optimized | 0.006x | 0.007x | True |
| copies_identity n=2 seed=0 | native | 0.006x | 0.008x | True |
| copies_identity n=8 seed=0 | optimized | 0.021x | 0.016x | False |
| copies_identity n=8 seed=0 | native | 0.024x | 0.021x | False |
| identity n=2 seed=0 | optimized | 0.005x | 0.007x | True |
| identity n=2 seed=0 | native | 0.004x | 0.005x | True |
| copies_lcomp n=2 seed=0 | optimized | 0.019x | 0.013x | True |
| copies_lcomp n=2 seed=0 | native | 0.031x | 0.017x | True |
| copies_lcomp n=8 seed=0 | optimized | 0.058x | 0.030x | True |
| copies_lcomp n=8 seed=0 | native | 0.101x | 0.039x | True |
| lcomp n=2 seed=0 | optimized | 0.012x | 0.011x | True |
| lcomp n=2 seed=0 | native | 0.015x | 0.010x | True |
| copies_pivot n=2 seed=0 | optimized | 0.015x | 0.014x | True |
| copies_pivot n=2 seed=0 | native | 0.035x | 0.021x | True |
| copies_pivot n=8 seed=0 | optimized | 0.051x | 0.036x | False |
| copies_pivot n=8 seed=0 | native | 0.076x | 0.036x | False |
| pivot n=2 seed=0 | optimized | 0.007x | 0.009x | True |
| pivot n=2 seed=0 | native | 0.020x | 0.014x | True |
| copies_pivot_gadget n=2 seed=0 | optimized | 0.037x | 0.024x | True |
| copies_pivot_gadget n=2 seed=0 | native | 0.071x | 0.030x | True |
| copies_pivot_gadget n=8 seed=0 | optimized | 0.080x | 0.043x | False |
| copies_pivot_gadget n=8 seed=0 | native | 0.183x | 0.056x | False |
| pivot_gadget n=2 seed=0 | optimized | 0.016x | 0.013x | True |
| pivot_gadget n=2 seed=0 | native | 0.038x | 0.018x | True |
| copies_pivot_boundary n=2 seed=0 | optimized | 0.041x | 0.026x | True |
| copies_pivot_boundary n=2 seed=0 | native | 0.050x | 0.022x | True |
| copies_pivot_boundary n=8 seed=0 | optimized | 0.063x | 0.032x | False |
| copies_pivot_boundary n=8 seed=0 | native | 0.122x | 0.041x | False |
| pivot_boundary n=2 seed=0 | optimized | 0.014x | 0.011x | True |
| pivot_boundary n=2 seed=0 | native | 0.030x | 0.015x | True |
| copies_gadget_fusion n=2 seed=0 | optimized | 0.004x | 0.007x | True |
| copies_gadget_fusion n=2 seed=0 | native | 0.007x | 0.010x | True |
| copies_gadget_fusion n=8 seed=0 | optimized | 0.020x | 0.024x | True |
| copies_gadget_fusion n=8 seed=0 | native | 0.020x | 0.023x | True |
| gadget_fusion n=2 seed=0 | optimized | 0.004x | 0.006x | True |
| gadget_fusion n=2 seed=0 | native | 0.006x | 0.010x | True |
| copies_bialgebra n=2 seed=0 | optimized | 0.036x | 0.025x | True |
| copies_bialgebra n=2 seed=0 | native | 0.034x | 0.024x | True |
| copies_bialgebra n=8 seed=0 | optimized | 0.030x | 0.025x | False |
| copies_bialgebra n=8 seed=0 | native | 0.028x | 0.024x | False |
| bialgebra n=2 seed=0 | optimized | 0.015x | 0.011x | True |
| bialgebra n=2 seed=0 | native | 0.021x | 0.014x | True |
| copies_hadamard_cancel n=2 seed=0 | optimized | 0.004x | 0.006x | True |
| copies_hadamard_cancel n=2 seed=0 | native | 0.004x | 0.006x | True |
| copies_hadamard_cancel n=8 seed=0 | optimized | 0.004x | 0.006x | False |
| copies_hadamard_cancel n=8 seed=0 | native | 0.004x | 0.006x | False |
| hadamard_cancel n=2 seed=0 | optimized | 0.003x | 0.005x | True |
| hadamard_cancel n=2 seed=0 | native | 0.003x | 0.005x | True |
| copies_supplementarity n=2 seed=0 | optimized | 0.007x | 0.010x | True |
| copies_supplementarity n=2 seed=0 | native | 0.041x | 0.020x | True |
| copies_supplementarity n=8 seed=0 | optimized | 0.004x | 0.008x | True |
| copies_supplementarity n=8 seed=0 | native | 0.074x | 0.048x | True |
| supplementarity n=2 seed=0 | optimized | 0.005x | 0.007x | True |
| supplementarity n=2 seed=0 | native | 0.017x | 0.012x | True |
| copies_copy n=2 seed=0 | optimized | 0.018x | 0.016x | True |
| copies_copy n=2 seed=0 | native | 0.012x | 0.012x | True |
| copies_copy n=8 seed=0 | optimized | 0.016x | 0.018x | True |
| copies_copy n=8 seed=0 | native | 0.013x | 0.016x | True |
| copy n=2 seed=0 | optimized | 0.010x | 0.010x | True |
| copy n=2 seed=0 | native | 0.009x | 0.010x | True |
| random_clifford n=3 seed=0 | optimized | 0.013x | 0.013x | True |
| random_clifford n=3 seed=0 | native | 0.022x | 0.022x | True |
| random_clifford_t n=3 seed=0 | optimized | 0.012x | 0.013x | True |
| random_clifford_t n=3 seed=0 | native | 0.018x | 0.018x | True |
| random_t_heavy n=3 seed=0 | optimized | 0.013x | 0.014x | True |
| random_t_heavy n=3 seed=0 | native | 0.015x | 0.015x | True |
| layered_local n=3 seed=0 | optimized | 0.013x | 0.014x | True |
| layered_local n=3 seed=0 | native | 0.030x | 0.028x | True |
| layered_global n=3 seed=0 | optimized | 0.017x | 0.016x | True |
| layered_global n=3 seed=0 | native | 0.021x | 0.020x | True |

## Failures and unresolved evidence


## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking and legacy JSON import/export remain audit concerns. See [audit history](../../../evaluation/audit-history.md).
