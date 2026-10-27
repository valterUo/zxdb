# ZXdb evaluation

Run state: finished. Profile: full.

## Correctness

{'verified': 435, 'inconclusive': 6, 'error': 1, 'failed': 4}

Each engine is checked against the original ordered tensor, up to a nonzero global scalar. Zero/nonzero disagreement fails. Tensor resource limits are inconclusive, never passes.

## Performance

209/222 performance cells completed without failures. 10 rewrite wins; 6 total-time wins. Inspect tensor evidence per cell before drawing conclusions.

Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; ranges are observed variation, not confidence intervals. Warmups excluded. Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.

Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter's in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.

| Workload | Variant | Rewrite speedup | Total speedup | Tensor verified |
|---|---|---:|---:|---|
| copies_spider_fusion n=2 seed=0 | control | 0.023x | 0.021x | True |
| copies_spider_fusion n=2 seed=0 | optimized | 0.016x | 0.015x | True |
| copies_spider_fusion n=32 seed=0 | control | 0.252x | 0.102x | False |
| copies_spider_fusion n=32 seed=0 | optimized | 0.186x | 0.081x | False |
| copies_spider_fusion n=512 seed=0 | control | 11.288x | 1.051x | False |
| copies_spider_fusion n=512 seed=0 | optimized | 12.245x | 1.127x | False |
| spider_fusion n=2 seed=0 | control | 0.004x | 0.005x | True |
| spider_fusion n=2 seed=0 | optimized | 0.005x | 0.005x | True |
| spider_fusion n=16 seed=0 | control | 0.023x | 0.015x | False |
| spider_fusion n=16 seed=0 | optimized | 0.046x | 0.029x | False |
| spider_fusion n=128 seed=0 | control | 0.431x | 0.097x | False |
| spider_fusion n=128 seed=0 | optimized | — | — | False |
| copies_identity n=2 seed=0 | control | 0.014x | 0.014x | True |
| copies_identity n=2 seed=0 | optimized | 0.018x | 0.015x | True |
| copies_identity n=32 seed=0 | control | 0.557x | 0.142x | False |
| copies_identity n=32 seed=0 | optimized | 0.646x | 0.127x | False |
| copies_identity n=512 seed=0 | control | 20.135x | 1.106x | False |
| copies_identity n=512 seed=0 | optimized | 22.854x | 1.263x | False |
| identity n=2 seed=0 | control | 0.005x | 0.005x | True |
| identity n=2 seed=0 | optimized | 0.003x | 0.004x | True |
| identity n=16 seed=0 | control | 0.021x | 0.015x | False |
| identity n=16 seed=0 | optimized | 0.020x | 0.012x | False |
| identity n=128 seed=0 | control | 0.419x | 0.077x | False |
| identity n=128 seed=0 | optimized | 0.873x | 0.120x | False |
| copies_lcomp n=2 seed=0 | control | 0.008x | 0.007x | True |
| copies_lcomp n=2 seed=0 | optimized | 0.010x | 0.008x | True |
| copies_lcomp n=32 seed=0 | control | 0.010x | 0.012x | False |
| copies_lcomp n=32 seed=0 | optimized | 0.130x | 0.055x | False |
| copies_lcomp n=512 seed=0 | control | 0.083x | 0.075x | False |
| copies_lcomp n=512 seed=0 | optimized | 6.428x | 0.430x | False |
| lcomp n=2 seed=0 | control | 0.009x | 0.008x | True |
| lcomp n=2 seed=0 | optimized | 0.009x | 0.008x | True |
| lcomp n=16 seed=0 | control | 0.033x | 0.015x | False |
| lcomp n=16 seed=0 | optimized | 0.032x | 0.019x | False |
| lcomp n=128 seed=0 | control | 0.036x | 0.018x | False |
| lcomp n=128 seed=0 | optimized | 0.116x | 0.034x | False |
| copies_pivot n=2 seed=0 | control | 0.014x | 0.012x | True |
| copies_pivot n=2 seed=0 | optimized | 0.015x | 0.012x | True |
| copies_pivot n=32 seed=0 | control | 0.030x | 0.025x | False |
| copies_pivot n=32 seed=0 | optimized | 0.205x | 0.043x | False |
| copies_pivot n=512 seed=0 | control | — | — | False |
| copies_pivot n=512 seed=0 | optimized | 6.334x | 0.590x | False |
| pivot n=2 seed=0 | control | 0.011x | 0.010x | True |
| pivot n=2 seed=0 | optimized | 0.010x | 0.009x | True |
| pivot n=16 seed=0 | control | 0.058x | 0.028x | False |
| pivot n=16 seed=0 | optimized | 0.051x | 0.022x | False |
| pivot n=128 seed=0 | control | 0.621x | 0.065x | False |
| pivot n=128 seed=0 | optimized | 0.656x | 0.083x | False |
| copies_pivot_gadget n=2 seed=0 | control | 0.018x | 0.014x | True |
| copies_pivot_gadget n=2 seed=0 | optimized | 0.016x | 0.012x | True |
| copies_pivot_gadget n=32 seed=0 | control | 0.023x | 0.020x | False |
| copies_pivot_gadget n=32 seed=0 | optimized | 0.148x | 0.034x | False |
| copies_pivot_gadget n=512 seed=0 | control | 0.051x | 0.062x | False |
| copies_pivot_gadget n=512 seed=0 | optimized | — | — | False |
| pivot_gadget n=2 seed=0 | control | 0.037x | 0.029x | True |
| pivot_gadget n=2 seed=0 | optimized | 0.036x | 0.025x | True |
| pivot_gadget n=16 seed=0 | control | 0.189x | 0.049x | False |
| pivot_gadget n=16 seed=0 | optimized | 0.216x | 0.061x | False |
| pivot_gadget n=128 seed=0 | control | 9.922x | 0.372x | False |
| pivot_gadget n=128 seed=0 | optimized | 6.503x | 0.515x | False |
| copies_pivot_boundary n=2 seed=0 | control | 0.019x | 0.015x | True |
| copies_pivot_boundary n=2 seed=0 | optimized | 0.036x | 0.021x | True |
| copies_pivot_boundary n=32 seed=0 | control | 0.022x | 0.023x | False |
| copies_pivot_boundary n=32 seed=0 | optimized | 0.354x | 0.054x | False |
| copies_pivot_boundary n=512 seed=0 | control | — | — | False |
| copies_pivot_boundary n=512 seed=0 | optimized | 0.585x | 0.073x | False |
| pivot_boundary n=2 seed=0 | control | 0.008x | 0.007x | True |
| pivot_boundary n=2 seed=0 | optimized | 0.010x | 0.008x | True |
| pivot_boundary n=16 seed=0 | control | 0.028x | 0.016x | False |
| pivot_boundary n=16 seed=0 | optimized | 0.040x | 0.019x | False |
| pivot_boundary n=128 seed=0 | control | 0.129x | 0.023x | False |
| pivot_boundary n=128 seed=0 | optimized | 0.208x | 0.027x | False |
| copies_gadget_fusion n=2 seed=0 | control | 0.011x | 0.013x | True |
| copies_gadget_fusion n=2 seed=0 | optimized | 0.007x | 0.011x | True |
| copies_gadget_fusion n=32 seed=0 | control | 0.049x | 0.032x | False |
| copies_gadget_fusion n=32 seed=0 | optimized | 0.044x | 0.033x | False |
| copies_gadget_fusion n=512 seed=0 | control | 0.527x | 0.059x | False |
| copies_gadget_fusion n=512 seed=0 | optimized | 0.410x | 0.057x | False |
| gadget_fusion n=2 seed=0 | control | 0.002x | 0.003x | True |
| gadget_fusion n=2 seed=0 | optimized | 0.002x | 0.003x | True |
| gadget_fusion n=16 seed=0 | control | 0.004x | 0.008x | True |
| gadget_fusion n=16 seed=0 | optimized | 0.006x | 0.010x | True |
| gadget_fusion n=128 seed=0 | control | 0.031x | 0.047x | False |
| gadget_fusion n=128 seed=0 | optimized | 0.017x | 0.033x | False |
| copies_bialgebra n=2 seed=0 | control | 0.011x | 0.008x | True |
| copies_bialgebra n=2 seed=0 | optimized | 0.012x | 0.009x | True |
| copies_bialgebra n=32 seed=0 | control | 0.014x | 0.014x | False |
| copies_bialgebra n=32 seed=0 | optimized | 0.008x | 0.008x | False |
| copies_bialgebra n=512 seed=0 | control | — | — | False |
| copies_bialgebra n=512 seed=0 | optimized | 0.012x | 0.012x | False |
| bialgebra n=2 seed=0 | control | 0.017x | 0.012x | True |
| bialgebra n=2 seed=0 | optimized | 0.023x | 0.016x | True |
| bialgebra n=16 seed=0 | control | 0.626x | 0.219x | False |
| bialgebra n=16 seed=0 | optimized | 0.577x | 0.205x | False |
| bialgebra n=128 seed=0 | control | 1.852x | 1.335x | False |
| bialgebra n=128 seed=0 | optimized | 1.577x | 1.200x | False |
| copies_hadamard_cancel n=2 seed=0 | control | 0.003x | 0.005x | True |
| copies_hadamard_cancel n=2 seed=0 | optimized | 0.003x | 0.005x | True |
| copies_hadamard_cancel n=32 seed=0 | control | 0.008x | 0.011x | False |
| copies_hadamard_cancel n=32 seed=0 | optimized | 0.011x | 0.013x | False |
| copies_hadamard_cancel n=512 seed=0 | control | 0.026x | 0.026x | False |
| copies_hadamard_cancel n=512 seed=0 | optimized | 0.035x | 0.035x | False |
| hadamard_cancel n=2 seed=0 | control | 0.003x | 0.005x | True |
| hadamard_cancel n=2 seed=0 | optimized | 0.004x | 0.007x | True |
| hadamard_cancel n=16 seed=0 | control | 0.005x | 0.007x | False |
| hadamard_cancel n=16 seed=0 | optimized | 0.005x | 0.007x | False |
| hadamard_cancel n=128 seed=0 | control | 0.017x | 0.016x | False |
| hadamard_cancel n=128 seed=0 | optimized | 0.014x | 0.015x | False |
| copies_supplementarity n=2 seed=0 | control | 0.005x | 0.008x | True |
| copies_supplementarity n=2 seed=0 | optimized | 0.005x | 0.007x | True |
| copies_supplementarity n=32 seed=0 | control | 0.001x | 0.001x | False |
| copies_supplementarity n=32 seed=0 | optimized | 0.001x | 0.003x | False |
| copies_supplementarity n=512 seed=0 | control | — | — | False |
| copies_supplementarity n=512 seed=0 | optimized | — | — | False |
| supplementarity n=2 seed=0 | control | 0.007x | 0.008x | True |
| supplementarity n=2 seed=0 | optimized | 0.009x | 0.012x | True |
| supplementarity n=16 seed=0 | control | 0.018x | 0.021x | True |
| supplementarity n=16 seed=0 | optimized | 0.011x | 0.015x | True |
| supplementarity n=128 seed=0 | control | 0.023x | 0.034x | False |
| supplementarity n=128 seed=0 | optimized | 0.016x | 0.025x | False |
| copies_copy n=2 seed=0 | control | 0.017x | 0.015x | True |
| copies_copy n=2 seed=0 | optimized | 0.024x | 0.022x | True |
| copies_copy n=32 seed=0 | control | 0.027x | 0.032x | False |
| copies_copy n=32 seed=0 | optimized | 0.014x | 0.018x | False |
| copies_copy n=512 seed=0 | control | 0.012x | 0.018x | False |
| copies_copy n=512 seed=0 | optimized | — | — | False |
| copy n=2 seed=0 | control | 0.019x | 0.015x | True |
| copy n=2 seed=0 | optimized | 0.017x | 0.013x | True |
| copy n=16 seed=0 | control | 0.038x | 0.027x | True |
| copy n=16 seed=0 | optimized | 0.037x | 0.020x | True |
| copy n=128 seed=0 | control | 0.247x | 0.052x | False |
| copy n=128 seed=0 | optimized | 0.156x | 0.045x | False |
| random_clifford n=3 seed=0 | control | 0.011x | 0.011x | True |
| random_clifford n=3 seed=0 | optimized | 0.033x | 0.032x | True |
| random_clifford n=3 seed=1 | control | 0.024x | 0.024x | True |
| random_clifford n=3 seed=1 | optimized | 0.020x | 0.019x | True |
| random_clifford n=3 seed=2 | control | 0.023x | 0.023x | True |
| random_clifford n=3 seed=2 | optimized | 0.022x | 0.023x | True |
| random_clifford n=8 seed=0 | control | 0.060x | 0.057x | False |
| random_clifford n=8 seed=0 | optimized | 0.088x | 0.082x | False |
| random_clifford n=8 seed=1 | control | 0.075x | 0.072x | False |
| random_clifford n=8 seed=1 | optimized | — | — | False |
| random_clifford n=8 seed=2 | control | 0.091x | 0.085x | False |
| random_clifford n=8 seed=2 | optimized | 0.103x | 0.095x | False |
| random_clifford n=16 seed=0 | control | 0.357x | 0.303x | False |
| random_clifford n=16 seed=0 | optimized | 0.143x | 0.132x | False |
| random_clifford n=16 seed=1 | control | 0.224x | 0.200x | False |
| random_clifford n=16 seed=1 | optimized | 0.133x | 0.124x | False |
| random_clifford n=16 seed=2 | control | 0.321x | 0.267x | False |
| random_clifford n=16 seed=2 | optimized | — | — | False |
| random_clifford_t n=3 seed=0 | control | 0.020x | 0.021x | True |
| random_clifford_t n=3 seed=0 | optimized | 0.024x | 0.025x | True |
| random_clifford_t n=3 seed=1 | control | 0.020x | 0.020x | True |
| random_clifford_t n=3 seed=1 | optimized | 0.031x | 0.031x | True |
| random_clifford_t n=3 seed=2 | control | 0.016x | 0.016x | True |
| random_clifford_t n=3 seed=2 | optimized | 0.016x | 0.017x | True |
| random_clifford_t n=8 seed=0 | control | 0.052x | 0.050x | False |
| random_clifford_t n=8 seed=0 | optimized | 0.058x | 0.055x | False |
| random_clifford_t n=8 seed=1 | control | 0.047x | 0.045x | False |
| random_clifford_t n=8 seed=1 | optimized | 0.085x | 0.079x | False |
| random_clifford_t n=8 seed=2 | control | 0.038x | 0.037x | False |
| random_clifford_t n=8 seed=2 | optimized | 0.038x | 0.038x | False |
| random_clifford_t n=16 seed=0 | control | 0.111x | 0.098x | False |
| random_clifford_t n=16 seed=0 | optimized | 0.140x | 0.124x | False |
| random_clifford_t n=16 seed=1 | control | 0.097x | 0.089x | False |
| random_clifford_t n=16 seed=1 | optimized | 0.121x | 0.113x | False |
| random_clifford_t n=16 seed=2 | control | 0.059x | 0.057x | False |
| random_clifford_t n=16 seed=2 | optimized | 0.067x | 0.064x | False |
| random_t_heavy n=3 seed=0 | control | 0.005x | 0.006x | True |
| random_t_heavy n=3 seed=0 | optimized | 0.013x | 0.014x | True |
| random_t_heavy n=3 seed=1 | control | 0.005x | 0.005x | True |
| random_t_heavy n=3 seed=1 | optimized | 0.011x | 0.011x | True |
| random_t_heavy n=3 seed=2 | control | 0.005x | 0.005x | True |
| random_t_heavy n=3 seed=2 | optimized | 0.006x | 0.007x | True |
| random_t_heavy n=8 seed=0 | control | 0.017x | 0.018x | True |
| random_t_heavy n=8 seed=0 | optimized | 0.024x | 0.024x | True |
| random_t_heavy n=8 seed=1 | control | — | — | False |
| random_t_heavy n=8 seed=1 | optimized | 0.074x | 0.072x | True |
| random_t_heavy n=8 seed=2 | control | 0.023x | 0.024x | True |
| random_t_heavy n=8 seed=2 | optimized | 0.043x | 0.041x | False |
| random_t_heavy n=16 seed=0 | control | 0.050x | 0.049x | False |
| random_t_heavy n=16 seed=0 | optimized | 0.067x | 0.063x | False |
| random_t_heavy n=16 seed=1 | control | 0.062x | 0.058x | False |
| random_t_heavy n=16 seed=1 | optimized | 0.087x | 0.081x | False |
| random_t_heavy n=16 seed=2 | control | 0.067x | 0.064x | False |
| random_t_heavy n=16 seed=2 | optimized | 0.071x | 0.068x | False |
| layered_local n=3 seed=0 | control | 0.024x | 0.023x | True |
| layered_local n=3 seed=0 | optimized | — | — | False |
| layered_local n=3 seed=1 | control | 0.030x | 0.030x | True |
| layered_local n=3 seed=1 | optimized | 0.036x | 0.034x | True |
| layered_local n=3 seed=2 | control | 0.021x | 0.022x | True |
| layered_local n=3 seed=2 | optimized | 0.039x | 0.037x | True |
| layered_local n=8 seed=0 | control | 0.068x | 0.063x | False |
| layered_local n=8 seed=0 | optimized | 0.052x | 0.046x | False |
| layered_local n=8 seed=1 | control | 0.044x | 0.042x | False |
| layered_local n=8 seed=1 | optimized | 0.054x | 0.051x | False |
| layered_local n=8 seed=2 | control | 0.047x | 0.045x | False |
| layered_local n=8 seed=2 | optimized | 0.060x | 0.055x | False |
| layered_local n=16 seed=0 | control | 0.111x | 0.101x | False |
| layered_local n=16 seed=0 | optimized | — | — | False |
| layered_local n=16 seed=1 | control | 0.105x | 0.095x | False |
| layered_local n=16 seed=1 | optimized | 0.101x | 0.093x | False |
| layered_local n=16 seed=2 | control | 0.090x | 0.082x | False |
| layered_local n=16 seed=2 | optimized | 0.102x | 0.095x | False |
| layered_global n=3 seed=0 | control | 0.012x | 0.012x | True |
| layered_global n=3 seed=0 | optimized | 0.017x | 0.017x | True |
| layered_global n=3 seed=1 | control | 0.010x | 0.010x | True |
| layered_global n=3 seed=1 | optimized | 0.015x | 0.015x | True |
| layered_global n=3 seed=2 | control | 0.010x | 0.010x | True |
| layered_global n=3 seed=2 | optimized | 0.022x | 0.022x | True |
| layered_global n=8 seed=0 | control | 0.075x | 0.070x | False |
| layered_global n=8 seed=0 | optimized | 0.073x | 0.069x | False |
| layered_global n=8 seed=1 | control | 0.073x | 0.070x | False |
| layered_global n=8 seed=1 | optimized | 0.101x | 0.092x | False |
| layered_global n=8 seed=2 | control | 0.074x | 0.069x | False |
| layered_global n=8 seed=2 | optimized | 0.103x | 0.094x | False |
| layered_global n=16 seed=0 | control | 0.220x | 0.197x | False |
| layered_global n=16 seed=0 | optimized | 0.226x | 0.203x | False |
| layered_global n=16 seed=1 | control | 0.192x | 0.176x | False |
| layered_global n=16 seed=1 | optimized | 0.194x | 0.177x | False |
| layered_global n=16 seed=2 | control | 0.162x | 0.149x | False |
| layered_global n=16 seed=2 | optimized | 0.254x | 0.219x | False |

## Failures and unresolved evidence

- regressions /  / unittest / : Regression subprocess exit code 4294967295; see regressions.log.
- correctness / full_reduce / random_q4_d10_s1034 / optimized: TimeoutError: Trial exceeded 60s; database interrupted
- correctness / full_reduce / isolated_zero_spider / control: {'db_check': {'status': 'failed', 'reason': 'zero-map comparison'}, 'pyzx_check': {'status': 'verified', 'reason': 'zero-map comparison'}}
- correctness / full_reduce / isolated_zero_spider / optimized: {'db_check': {'status': 'failed', 'reason': 'zero-map comparison'}, 'pyzx_check': {'status': 'verified', 'reason': 'zero-map comparison'}}
- correctness / full_reduce / explicit_zero_scalar / control: {'db_check': {'status': 'failed', 'reason': 'zero-map comparison'}, 'pyzx_check': {'status': 'verified', 'reason': 'zero-map comparison'}}
- correctness / full_reduce / explicit_zero_scalar / optimized: {'db_check': {'status': 'failed', 'reason': 'zero-map comparison'}, 'pyzx_check': {'status': 'verified', 'reason': 'zero-map comparison'}}
- benchmark / spider_fusion / f18a23ffde6e4067 / optimized: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / pivot / d3274a782d48a83a / control: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / pivot_gadget / 3bf9c84ebd22d67c / optimized: TimeoutError: Trial exceeded 60s; database interrupted
- benchmark / pivot_boundary / f67e1b127d8dea02 / control: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- pipeline /  / execution / : subprocess.CalledProcessError: Command '['docker', 'restart', 'memgraph']' returned non-zero exit status 1.
- pipeline /  / cleanup / : Couldn't connect to localhost:7687 (resolved to ('[::1]:7687', '127.0.0.1:7687')):
- benchmark / bialgebra / 6005997c57742351 / control: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / supplementarity / f2b3998d0a92a661 / control: TimeoutError: Trial exceeded 60s; database interrupted
- benchmark / supplementarity / f2b3998d0a92a661 / optimized: TimeoutError: Trial exceeded 60s; database interrupted
- benchmark / copy / 2d62f66671acacf2 / optimized: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / full_reduce / d49665188a4dbe47 / optimized: TimeoutError: Trial exceeded 60s; database interrupted
- benchmark / full_reduce / cf71e9c5cb9d5703 / optimized: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / full_reduce / cdd18adf691ffe41 / control: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / full_reduce / 74a602923d5bb6d2 / optimized: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))
- benchmark / full_reduce / a64996c0bb9c6ad1 / optimized: ServiceUnavailable: Failed to read from defunct connection IPv4Address(('localhost', 7687)) (ResolvedIPv6Address(('::1', 7687, 0, 0)))

## Verification review

48 observations received a conservative evidence review after the absolute-zero threshold was found to misclassify tiny nonzero tensors (the 128-gadget input has maximum amplitude 5.42e-20). Their previous checks are retained in verification_before_review; ambiguous zero/nonfinite conclusions are now inconclusive, not failures or passes. No timing was changed or rerun. The two explicit zero-map counterexamples remain failures. The corrected checker requires exact zero factors, treats ambiguous near-zero/nonfinite tensors as inconclusive, and scales before taking norms. Six oracle/measurement unit tests pass. The regression subprocess in this measurement run was manually stopped after the database crashed; an earlier complete 52-test run passed, preserved in the backup. These outcomes do not establish general correctness or stability. The corrected checker also evaluated all 223 PyZX reference outputs: 220 verified, three inconclusive, none failed (reference-checks.json). Five targeted live checks confirmed the four explicit zero-map failures and one near-zero inconclusive result (verification-checks.json).

## Scope and interpretation

Control is the ordinary Cypher implementation; optimized uses independent rewrite batches with exclusive access and no incremental worklist. Hadamard cancellation uses the same implementation in both variants. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.

Large cases may be unverified even when their small motifs pass. No capacity or correctness claim follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes are saved beside this report. Historical summaries are separate and are not evidence for this revision.

Known limitations: database scalar tracking, legacy JSON import/export, and standalone supplementarity preconditions remain audit concerns. See ../../evaluation/audit-history.md.
