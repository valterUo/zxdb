# ZXdb evaluation

Run state: finished. Profile: full.

## Correctness

{'verified': 440, 'inconclusive': 6}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. Proofs run outside the timing/watchdog region and are cached by input/output hashes.

## Performance

222/222 performance cells completed without failures. 16 rewrite wins; 6 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |
|---|---|---:|---:|---|
| copies_spider_fusion n=2 seed=0 | optimized | 0.015x | 0.014x | True |
| copies_spider_fusion n=2 seed=0 | native | 0.018x | 0.017x | True |
| copies_spider_fusion n=32 seed=0 | optimized | 0.266x | 0.113x | False |
| copies_spider_fusion n=32 seed=0 | native | 0.322x | 0.118x | False |
| copies_spider_fusion n=512 seed=0 | optimized | 11.013x | 1.288x | False |
| copies_spider_fusion n=512 seed=0 | native | 10.074x | 1.392x | False |
| spider_fusion n=2 seed=0 | optimized | 0.002x | 0.003x | True |
| spider_fusion n=2 seed=0 | native | 0.011x | 0.013x | True |
| spider_fusion n=16 seed=0 | optimized | 0.054x | 0.038x | False |
| spider_fusion n=16 seed=0 | native | 0.073x | 0.045x | False |
| spider_fusion n=128 seed=0 | optimized | 0.802x | 0.206x | False |
| spider_fusion n=128 seed=0 | native | 0.942x | 0.245x | False |
| copies_identity n=2 seed=0 | optimized | 0.009x | 0.011x | True |
| copies_identity n=2 seed=0 | native | 0.009x | 0.010x | True |
| copies_identity n=32 seed=0 | optimized | 0.226x | 0.079x | False |
| copies_identity n=32 seed=0 | native | 0.178x | 0.078x | False |
| copies_identity n=512 seed=0 | optimized | 16.350x | 1.629x | False |
| copies_identity n=512 seed=0 | native | 14.702x | 1.418x | False |
| identity n=2 seed=0 | optimized | 0.002x | 0.004x | True |
| identity n=2 seed=0 | native | 0.003x | 0.003x | True |
| identity n=16 seed=0 | optimized | 0.018x | 0.014x | False |
| identity n=16 seed=0 | native | 0.016x | 0.015x | False |
| identity n=128 seed=0 | optimized | 0.439x | 0.145x | False |
| identity n=128 seed=0 | native | 0.391x | 0.139x | False |
| copies_lcomp n=2 seed=0 | optimized | 0.012x | 0.012x | True |
| copies_lcomp n=2 seed=0 | native | 0.021x | 0.012x | True |
| copies_lcomp n=32 seed=0 | optimized | 0.124x | 0.057x | False |
| copies_lcomp n=32 seed=0 | native | 0.222x | 0.062x | False |
| copies_lcomp n=512 seed=0 | optimized | 1.276x | 0.460x | False |
| copies_lcomp n=512 seed=0 | native | 7.330x | 0.653x | False |
| lcomp n=2 seed=0 | optimized | 0.004x | 0.004x | True |
| lcomp n=2 seed=0 | native | 0.009x | 0.006x | True |
| lcomp n=16 seed=0 | optimized | 0.018x | 0.013x | False |
| lcomp n=16 seed=0 | native | 0.029x | 0.014x | False |
| lcomp n=128 seed=0 | optimized | 0.236x | 0.025x | False |
| lcomp n=128 seed=0 | native | 0.306x | 0.027x | False |
| copies_pivot n=2 seed=0 | optimized | 0.006x | 0.007x | True |
| copies_pivot n=2 seed=0 | native | 0.016x | 0.010x | True |
| copies_pivot n=32 seed=0 | optimized | 0.102x | 0.055x | False |
| copies_pivot n=32 seed=0 | native | 0.175x | 0.059x | False |
| copies_pivot n=512 seed=0 | optimized | 1.862x | 0.550x | False |
| copies_pivot n=512 seed=0 | native | 6.951x | 0.662x | False |
| pivot n=2 seed=0 | optimized | 0.004x | 0.005x | True |
| pivot n=2 seed=0 | native | 0.014x | 0.011x | True |
| pivot n=16 seed=0 | optimized | 0.036x | 0.024x | False |
| pivot n=16 seed=0 | native | 0.071x | 0.031x | False |
| pivot n=128 seed=0 | optimized | 1.043x | 0.075x | False |
| pivot n=128 seed=0 | native | 0.966x | 0.078x | False |
| copies_pivot_gadget n=2 seed=0 | optimized | 0.016x | 0.012x | True |
| copies_pivot_gadget n=2 seed=0 | native | 0.026x | 0.014x | True |
| copies_pivot_gadget n=32 seed=0 | optimized | 0.130x | 0.043x | False |
| copies_pivot_gadget n=32 seed=0 | native | 0.285x | 0.069x | False |
| copies_pivot_gadget n=512 seed=0 | optimized | 1.096x | 0.257x | False |
| copies_pivot_gadget n=512 seed=0 | native | 5.310x | 0.326x | False |
| pivot_gadget n=2 seed=0 | optimized | 0.009x | 0.008x | True |
| pivot_gadget n=2 seed=0 | native | 0.027x | 0.013x | True |
| pivot_gadget n=16 seed=0 | optimized | 0.080x | 0.032x | False |
| pivot_gadget n=16 seed=0 | native | 0.139x | 0.036x | False |
| pivot_gadget n=128 seed=0 | optimized | 7.061x | 0.336x | False |
| pivot_gadget n=128 seed=0 | native | 8.838x | 0.438x | False |
| copies_pivot_boundary n=2 seed=0 | optimized | 0.021x | 0.015x | True |
| copies_pivot_boundary n=2 seed=0 | native | 0.032x | 0.014x | True |
| copies_pivot_boundary n=32 seed=0 | optimized | 0.163x | 0.052x | False |
| copies_pivot_boundary n=32 seed=0 | native | 0.358x | 0.074x | False |
| copies_pivot_boundary n=512 seed=0 | optimized | 0.900x | 0.159x | False |
| copies_pivot_boundary n=512 seed=0 | native | 1.346x | 0.109x | False |
| pivot_boundary n=2 seed=0 | optimized | 0.013x | 0.010x | True |
| pivot_boundary n=2 seed=0 | native | 0.023x | 0.013x | True |
| pivot_boundary n=16 seed=0 | optimized | 0.055x | 0.024x | False |
| pivot_boundary n=16 seed=0 | native | 0.077x | 0.026x | False |
| pivot_boundary n=128 seed=0 | optimized | 0.640x | 0.037x | False |
| pivot_boundary n=128 seed=0 | native | 0.780x | 0.040x | False |
| copies_gadget_fusion n=2 seed=0 | optimized | 0.004x | 0.007x | True |
| copies_gadget_fusion n=2 seed=0 | native | 0.004x | 0.005x | True |
| copies_gadget_fusion n=32 seed=0 | optimized | 0.028x | 0.024x | False |
| copies_gadget_fusion n=32 seed=0 | native | 0.029x | 0.025x | False |
| copies_gadget_fusion n=512 seed=0 | optimized | 0.484x | 0.080x | False |
| copies_gadget_fusion n=512 seed=0 | native | 0.496x | 0.075x | False |
| gadget_fusion n=2 seed=0 | optimized | 0.002x | 0.003x | True |
| gadget_fusion n=2 seed=0 | native | 0.001x | 0.003x | True |
| gadget_fusion n=16 seed=0 | optimized | 0.004x | 0.008x | True |
| gadget_fusion n=16 seed=0 | native | 0.004x | 0.008x | True |
| gadget_fusion n=128 seed=0 | optimized | 0.022x | 0.032x | False |
| gadget_fusion n=128 seed=0 | native | 0.086x | 0.070x | False |
| copies_bialgebra n=2 seed=0 | optimized | 0.041x | 0.027x | True |
| copies_bialgebra n=2 seed=0 | native | 0.049x | 0.033x | True |
| copies_bialgebra n=32 seed=0 | optimized | 0.040x | 0.035x | False |
| copies_bialgebra n=32 seed=0 | native | 0.038x | 0.031x | False |
| copies_bialgebra n=512 seed=0 | optimized | 0.010x | 0.010x | False |
| copies_bialgebra n=512 seed=0 | native | 0.009x | 0.010x | False |
| bialgebra n=2 seed=0 | optimized | 0.033x | 0.021x | True |
| bialgebra n=2 seed=0 | native | 0.024x | 0.016x | True |
| bialgebra n=16 seed=0 | optimized | 0.513x | 0.162x | False |
| bialgebra n=16 seed=0 | native | 0.528x | 0.162x | False |
| bialgebra n=128 seed=0 | optimized | 1.899x | 1.356x | False |
| bialgebra n=128 seed=0 | native | 2.026x | 1.380x | False |
| copies_hadamard_cancel n=2 seed=0 | optimized | 0.004x | 0.008x | True |
| copies_hadamard_cancel n=2 seed=0 | native | 0.005x | 0.007x | True |
| copies_hadamard_cancel n=32 seed=0 | optimized | 0.009x | 0.010x | False |
| copies_hadamard_cancel n=32 seed=0 | native | 0.010x | 0.011x | False |
| copies_hadamard_cancel n=512 seed=0 | optimized | 0.031x | 0.031x | False |
| copies_hadamard_cancel n=512 seed=0 | native | 0.029x | 0.029x | False |
| hadamard_cancel n=2 seed=0 | optimized | 0.003x | 0.005x | True |
| hadamard_cancel n=2 seed=0 | native | 0.004x | 0.006x | True |
| hadamard_cancel n=16 seed=0 | optimized | 0.007x | 0.010x | False |
| hadamard_cancel n=16 seed=0 | native | 0.005x | 0.009x | False |
| hadamard_cancel n=128 seed=0 | optimized | 0.013x | 0.014x | False |
| hadamard_cancel n=128 seed=0 | native | 0.013x | 0.014x | False |
| copies_supplementarity n=2 seed=0 | optimized | 0.008x | 0.011x | True |
| copies_supplementarity n=2 seed=0 | native | 0.027x | 0.019x | True |
| copies_supplementarity n=32 seed=0 | optimized | 0.004x | 0.007x | False |
| copies_supplementarity n=32 seed=0 | native | 0.144x | 0.076x | False |
| copies_supplementarity n=512 seed=0 | optimized | 0.000x | 0.000x | False |
| copies_supplementarity n=512 seed=0 | native | 0.028x | 0.039x | False |
| supplementarity n=2 seed=0 | optimized | 0.008x | 0.010x | True |
| supplementarity n=2 seed=0 | native | 0.028x | 0.016x | True |
| supplementarity n=16 seed=0 | optimized | 0.014x | 0.020x | True |
| supplementarity n=16 seed=0 | native | 0.068x | 0.037x | True |
| supplementarity n=128 seed=0 | optimized | 0.051x | 0.060x | False |
| supplementarity n=128 seed=0 | native | 0.089x | 0.068x | False |
| copies_copy n=2 seed=0 | optimized | 0.016x | 0.015x | True |
| copies_copy n=2 seed=0 | native | 0.025x | 0.021x | True |
| copies_copy n=32 seed=0 | optimized | 0.017x | 0.025x | False |
| copies_copy n=32 seed=0 | native | 0.017x | 0.024x | False |
| copies_copy n=512 seed=0 | optimized | 0.013x | 0.016x | False |
| copies_copy n=512 seed=0 | native | 0.012x | 0.015x | False |
| copy n=2 seed=0 | optimized | 0.010x | 0.009x | True |
| copy n=2 seed=0 | native | 0.011x | 0.009x | True |
| copy n=16 seed=0 | optimized | 0.032x | 0.023x | True |
| copy n=16 seed=0 | native | 0.025x | 0.018x | True |
| copy n=128 seed=0 | optimized | 0.105x | 0.042x | False |
| copy n=128 seed=0 | native | 0.106x | 0.045x | False |
| random_clifford n=3 seed=0 | optimized | 0.012x | 0.012x | True |
| random_clifford n=3 seed=0 | native | 0.015x | 0.015x | True |
| random_clifford n=3 seed=1 | optimized | 0.013x | 0.013x | True |
| random_clifford n=3 seed=1 | native | 0.023x | 0.022x | True |
| random_clifford n=3 seed=2 | optimized | 0.015x | 0.015x | True |
| random_clifford n=3 seed=2 | native | 0.018x | 0.018x | True |
| random_clifford n=8 seed=0 | optimized | 0.060x | 0.057x | False |
| random_clifford n=8 seed=0 | native | 0.090x | 0.083x | False |
| random_clifford n=8 seed=1 | optimized | 0.063x | 0.060x | False |
| random_clifford n=8 seed=1 | native | 0.088x | 0.082x | False |
| random_clifford n=8 seed=2 | optimized | 0.050x | 0.048x | False |
| random_clifford n=8 seed=2 | native | 0.150x | 0.135x | False |
| random_clifford n=16 seed=0 | optimized | 0.389x | 0.339x | False |
| random_clifford n=16 seed=0 | native | 0.664x | 0.536x | False |
| random_clifford n=16 seed=1 | optimized | 0.279x | 0.252x | False |
| random_clifford n=16 seed=1 | native | 0.399x | 0.342x | False |
| random_clifford n=16 seed=2 | optimized | 0.318x | 0.284x | False |
| random_clifford n=16 seed=2 | native | 0.494x | 0.410x | False |
| random_clifford_t n=3 seed=0 | optimized | 0.010x | 0.010x | True |
| random_clifford_t n=3 seed=0 | native | 0.015x | 0.016x | True |
| random_clifford_t n=3 seed=1 | optimized | 0.015x | 0.015x | True |
| random_clifford_t n=3 seed=1 | native | 0.017x | 0.018x | True |
| random_clifford_t n=3 seed=2 | optimized | 0.012x | 0.013x | True |
| random_clifford_t n=3 seed=2 | native | 0.017x | 0.017x | True |
| random_clifford_t n=8 seed=0 | optimized | 0.027x | 0.027x | False |
| random_clifford_t n=8 seed=0 | native | 0.044x | 0.042x | False |
| random_clifford_t n=8 seed=1 | optimized | 0.041x | 0.039x | False |
| random_clifford_t n=8 seed=1 | native | 0.059x | 0.055x | False |
| random_clifford_t n=8 seed=2 | optimized | 0.027x | 0.027x | False |
| random_clifford_t n=8 seed=2 | native | 0.044x | 0.043x | False |
| random_clifford_t n=16 seed=0 | optimized | 0.067x | 0.063x | False |
| random_clifford_t n=16 seed=0 | native | 0.095x | 0.084x | False |
| random_clifford_t n=16 seed=1 | optimized | 0.117x | 0.104x | False |
| random_clifford_t n=16 seed=1 | native | 0.188x | 0.158x | False |
| random_clifford_t n=16 seed=2 | optimized | 0.097x | 0.088x | False |
| random_clifford_t n=16 seed=2 | native | 0.166x | 0.142x | False |
| random_t_heavy n=3 seed=0 | optimized | 0.015x | 0.015x | True |
| random_t_heavy n=3 seed=0 | native | 0.021x | 0.022x | True |
| random_t_heavy n=3 seed=1 | optimized | 0.013x | 0.014x | True |
| random_t_heavy n=3 seed=1 | native | 0.012x | 0.013x | True |
| random_t_heavy n=3 seed=2 | optimized | 0.010x | 0.011x | True |
| random_t_heavy n=3 seed=2 | native | 0.012x | 0.013x | True |
| random_t_heavy n=8 seed=0 | optimized | 0.034x | 0.033x | False |
| random_t_heavy n=8 seed=0 | native | 0.041x | 0.040x | True |
| random_t_heavy n=8 seed=1 | optimized | 0.027x | 0.030x | True |
| random_t_heavy n=8 seed=1 | native | 0.030x | 0.029x | True |
| random_t_heavy n=8 seed=2 | optimized | 0.027x | 0.027x | False |
| random_t_heavy n=8 seed=2 | native | 0.040x | 0.038x | False |
| random_t_heavy n=16 seed=0 | optimized | 0.040x | 0.039x | False |
| random_t_heavy n=16 seed=0 | native | 0.051x | 0.050x | False |
| random_t_heavy n=16 seed=1 | optimized | 0.055x | 0.051x | False |
| random_t_heavy n=16 seed=1 | native | 0.094x | 0.086x | False |
| random_t_heavy n=16 seed=2 | optimized | 0.056x | 0.054x | False |
| random_t_heavy n=16 seed=2 | native | 0.071x | 0.067x | False |
| layered_local n=3 seed=0 | optimized | 0.014x | 0.014x | True |
| layered_local n=3 seed=0 | native | 0.019x | 0.019x | True |
| layered_local n=3 seed=1 | optimized | 0.013x | 0.013x | True |
| layered_local n=3 seed=1 | native | 0.035x | 0.033x | True |
| layered_local n=3 seed=2 | optimized | 0.026x | 0.026x | True |
| layered_local n=3 seed=2 | native | 0.036x | 0.034x | True |
| layered_local n=8 seed=0 | optimized | 0.093x | 0.080x | False |
| layered_local n=8 seed=0 | native | 0.107x | 0.093x | False |
| layered_local n=8 seed=1 | optimized | 0.079x | 0.072x | False |
| layered_local n=8 seed=1 | native | 0.119x | 0.101x | False |
| layered_local n=8 seed=2 | optimized | 0.080x | 0.073x | False |
| layered_local n=8 seed=2 | native | 0.118x | 0.099x | False |
| layered_local n=16 seed=0 | optimized | 0.246x | 0.207x | False |
| layered_local n=16 seed=0 | native | 0.221x | 0.173x | False |
| layered_local n=16 seed=1 | optimized | 0.172x | 0.140x | False |
| layered_local n=16 seed=1 | native | 0.243x | 0.191x | False |
| layered_local n=16 seed=2 | optimized | 0.209x | 0.176x | False |
| layered_local n=16 seed=2 | native | 0.238x | 0.186x | False |
| layered_global n=3 seed=0 | optimized | 0.018x | 0.017x | True |
| layered_global n=3 seed=0 | native | 0.028x | 0.027x | True |
| layered_global n=3 seed=1 | optimized | 0.021x | 0.020x | True |
| layered_global n=3 seed=1 | native | 0.023x | 0.022x | True |
| layered_global n=3 seed=2 | optimized | 0.019x | 0.019x | True |
| layered_global n=3 seed=2 | native | 0.026x | 0.025x | True |
| layered_global n=8 seed=0 | optimized | 0.059x | 0.056x | False |
| layered_global n=8 seed=0 | native | 0.072x | 0.067x | False |
| layered_global n=8 seed=1 | optimized | 0.072x | 0.066x | False |
| layered_global n=8 seed=1 | native | 0.113x | 0.099x | False |
| layered_global n=8 seed=2 | optimized | 0.060x | 0.056x | False |
| layered_global n=8 seed=2 | native | 0.170x | 0.144x | False |
| layered_global n=16 seed=0 | optimized | 0.365x | 0.311x | False |
| layered_global n=16 seed=0 | native | 0.482x | 0.382x | False |
| layered_global n=16 seed=1 | optimized | 0.427x | 0.360x | False |
| layered_global n=16 seed=1 | native | 0.550x | 0.439x | False |
| layered_global n=16 seed=2 | optimized | 0.342x | 0.286x | False |
| layered_global n=16 seed=2 | native | 0.478x | 0.374x | False |

## Duplicate seed inputs

These seed settings generate identical input graphs; they are repeated measurements, not independent workloads.

- layered_local n=3: seeds [0, 1, 2]
- layered_local n=8: seeds [0, 1, 2]
- layered_local n=16: seeds [0, 1, 2]
- layered_global n=3: seeds [1, 2]

## Failures and unresolved evidence


## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access. Hadamard cancellation uses the same implementation in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking and legacy JSON import/export remain audit concerns. See [audit history](../../../evaluation/audit-history.md).
