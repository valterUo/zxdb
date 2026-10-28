# Post-audit performance comparison

Run status: PARTIAL — extended measurements are still running.

118 workload configurations, all eleven rule families, five circuit families, and identity chains.

Times are seconds. Each database cell has one warmup and three measured repetitions. The standard runner supplies two PyZX groups, each with one warmup and three measured samples; the six measured samples are pooled here. Large-circuit PyZX has one warmup and three measured samples, shared across database modes. Speedups in this report are ratios of median times. Positive before/after change means slower.

Cypher means the optimized Cypher-only schedule. Hybrid uses scoped C++ procedures for pivot, local complementation, gadget/boundary pivot, and supplementarity; other rules remain Cypher. The C++ source is unchanged by the audit. Before-fix database paths use the saved pre-audit Python and query sources on the same server and installed dependencies. [Source provenance](baseline-provenance.json) confirms these three rewrite sources match the previous completed Cypher campaign after line-ending normalization.

## Correctness and completeness

{'regressions:verified': 1, 'correctness:verified': 440, 'correctness:inconclusive': 6, 'benchmark:verified': 338, 'benchmark:inconclusive': 550}

Extended observations: {'inconclusive': 140}.

Inconclusive tensor checks are not semantic successes: large open tensors exceed resource bounds. Failed or incomplete cells have no reported median or speedup. No-op cases retain elapsed times but receive no speedup. Timings exclude verification and snapshot writing.

## Largest tested size in each family

Circuit rows use geometric means across seed medians (three seeds in the completed grid). Other rows have one deterministic input. These aggregates do not hide the individual rows below.

| Workload | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher change | Hybrid change |
|---|---:|---:|---:|---:|---:|---:|
| bialgebra | 128 | 2.77205 | 1.48699 | 1.42402 | — | — |
| copies_bialgebra | 512 | 0.124097 | 13.3899 | 13.3931 | — | — |
| copies_copy | 512 | 0.0258228 | 2.18637 | 2.21655 | — | — |
| copies_gadget_fusion | 512 | 0.0303779 | 0.0594098 | 0.0625635 | — | — |
| copies_hadamard_cancel | 512 | 0.464931 | 15.9699 | 15.8958 | — | — |
| copies_identity | 512 | 0.485855 | 0.0306649 | 0.0339245 | — | — |
| copies_lcomp | 512 | 0.193704 | 0.1535 | 0.0262012 | — | — |
| copies_pivot | 512 | 0.412317 | 0.23092 | 0.0578135 | — | — |
| copies_pivot_boundary | 512 | 0.074856 | 0.081232 | 0.0518072 | — | — |
| copies_pivot_gadget | 512 | 0.261387 | 0.228118 | 0.0503803 | — | — |
| copies_spider_fusion | 512 | 0.661853 | 0.0595188 | 0.0649129 | — | — |
| copies_supplementarity | 512 | 0.0136923 | 90.328 | 0.478154 | — | — |
| copy | 128 | 0.00092295 | 0.0087758 | 0.0090815 | — | — |
| gadget_fusion | 128 | 0.0008978 | 0.0407006 | 0.0104468 | — | — |
| hadamard_cancel | 128 | 0.00799055 | 0.617265 | 0.674913 | — | — |
| identity | 128 | 0.00942695 | 0.0224629 | 0.0253862 | — | — |
| layered_global | 16 | 0.0878332 | 0.227362 | 0.167662 | — | — |
| layered_local | 16 | 0.0381776 | 0.203267 | 0.153195 | — | — |
| lcomp | 128 | 0.00754595 | 0.0303236 | 0.0257786 | — | — |
| pivot | 128 | 0.0636114 | 0.0551715 | 0.0708494 | — | — |
| pivot_boundary | 128 | 0.0291256 | 0.0430272 | 0.0385787 | — | — |
| pivot_gadget | 128 | 0.335818 | 0.0407001 | 0.0410313 | — | — |
| random_clifford | 128 | 141.306 | 17.1721 | 16.9477 | -1.1% | -0.4% |
| random_clifford_t | 16 | 0.0291601 | 0.309326 | 0.205402 | — | — |
| random_t_heavy | 16 | 0.0188853 | 0.365805 | 0.280154 | — | — |
| spider_fusion | 128 | 0.01562 | 0.0157603 | 0.0187321 | — | — |
| supplementarity | 128 | 0.00180965 | 0.0326114 | 0.0204233 | — | — |

## All workload timings

| Workload | Size | Depth | Seed | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher change vs pre-fix | Hybrid change vs pre-fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bialgebra | 2 | — | 0 | 0.00027495 | 0.0076604 | 0.0101735 | — | — |
| bialgebra | 16 | — | 0 | 0.0072576 | 0.0142175 | 0.0133686 | — | — |
| bialgebra | 128 | — | 0 | 2.77205 | 1.48699 | 1.42402 | — | — |
| copies_bialgebra | 2 | — | 0 | 0.0005203 | 0.0112215 | 0.0128934 | — | — |
| copies_bialgebra | 32 | — | 0 | 0.00739755 | 0.199284 | 0.193909 | — | — |
| copies_bialgebra | 512 | — | 0 | 0.124097 | 13.3899 | 13.3931 | — | — |
| copies_copy | 2 | — | 0 | 0.0001415 | 0.0087303 | 0.0059715 | — | — |
| copies_copy | 32 | — | 0 | 0.0010774 | 0.0619171 | 0.0608464 | — | — |
| copies_copy | 512 | — | 0 | 0.0258228 | 2.18637 | 2.21655 | — | — |
| copies_gadget_fusion | 2 | — | 0 | 8.95e-05 | 0.023841 | 0.0234137 | — | — |
| copies_gadget_fusion | 32 | — | 0 | 0.00065245 | 0.023151 | 0.0246458 | — | — |
| copies_gadget_fusion | 512 | — | 0 | 0.0303779 | 0.0594098 | 0.0625635 | — | — |
| copies_hadamard_cancel | 2 | — | 0 | 9.48e-05 | 0.0199968 | 0.0230212 | — | — |
| copies_hadamard_cancel | 32 | — | 0 | 0.0023477 | 0.277186 | 0.284855 | — | — |
| copies_hadamard_cancel | 512 | — | 0 | 0.464931 | 15.9699 | 15.8958 | — | — |
| copies_identity | 2 | — | 0 | 0.00010305 | 0.0110324 | 0.0104394 | — | — |
| copies_identity | 32 | — | 0 | 0.0022053 | 0.0099487 | 0.0119881 | — | — |
| copies_identity | 512 | — | 0 | 0.485855 | 0.0306649 | 0.0339245 | — | — |
| copies_lcomp | 2 | — | 0 | 0.0001394 | 0.0127304 | 0.0064923 | — | — |
| copies_lcomp | 32 | — | 0 | 0.0019408 | 0.0148244 | 0.0094075 | — | — |
| copies_lcomp | 512 | — | 0 | 0.193704 | 0.1535 | 0.0262012 | — | — |
| copies_pivot | 2 | — | 0 | 0.0002377 | 0.037967 | 0.0151404 | — | — |
| copies_pivot | 32 | — | 0 | 0.0039859 | 0.044143 | 0.0205281 | — | — |
| copies_pivot | 512 | — | 0 | 0.412317 | 0.23092 | 0.0578135 | — | — |
| copies_pivot_boundary | 2 | — | 0 | 0.00031945 | 0.0152403 | 0.0079606 | — | — |
| copies_pivot_boundary | 32 | — | 0 | 0.0040662 | 0.022174 | 0.0111597 | — | — |
| copies_pivot_boundary | 512 | — | 0 | 0.074856 | 0.081232 | 0.0518072 | — | — |
| copies_pivot_gadget | 2 | — | 0 | 0.0002822 | 0.0166825 | 0.0112896 | — | — |
| copies_pivot_gadget | 32 | — | 0 | 0.003143 | 0.0207679 | 0.0142132 | — | — |
| copies_pivot_gadget | 512 | — | 0 | 0.261387 | 0.228118 | 0.0503803 | — | — |
| copies_spider_fusion | 2 | — | 0 | 0.00019405 | 0.0128841 | 0.0126962 | — | — |
| copies_spider_fusion | 32 | — | 0 | 0.0052313 | 0.0183469 | 0.0166503 | — | — |
| copies_spider_fusion | 512 | — | 0 | 0.661853 | 0.0595188 | 0.0649129 | — | — |
| copies_supplementarity | 2 | — | 0 | 0.0001535 | 0.018634 | 0.0038127 | — | — |
| copies_supplementarity | 32 | — | 0 | 0.00098295 | 0.244847 | 0.0070562 | — | — |
| copies_supplementarity | 512 | — | 0 | 0.0136923 | 90.328 | 0.478154 | — | — |
| copy | 2 | — | 0 | 6.47e-05 | 0.0060538 | 0.0055076 | — | — |
| copy | 16 | — | 0 | 0.000176 | 0.0056607 | 0.0061576 | — | — |
| copy | 128 | — | 0 | 0.00092295 | 0.0087758 | 0.0090815 | — | — |
| gadget_fusion | 2 | — | 0 | 6.48e-05 | 0.0412655 | 0.0415672 | — | — |
| gadget_fusion | 16 | — | 0 | 0.00017175 | 0.0400858 | 0.0401923 | — | — |
| gadget_fusion | 128 | — | 0 | 0.0008978 | 0.0407006 | 0.0104468 | — | — |
| hadamard_cancel | 2 | — | 0 | 4.465e-05 | 0.012885 | 0.0134693 | — | — |
| hadamard_cancel | 16 | — | 0 | 0.00036945 | 0.0684593 | 0.0581241 | — | — |
| hadamard_cancel | 128 | — | 0 | 0.00799055 | 0.617265 | 0.674913 | — | — |
| identity | 2 | — | 0 | 5.635e-05 | 0.0213361 | 0.023129 | — | — |
| identity | 16 | — | 0 | 0.00035145 | 0.0232139 | 0.0185165 | — | — |
| identity | 128 | — | 0 | 0.00942695 | 0.0224629 | 0.0253862 | — | — |
| layered_global | 3 | 2 | 0 | 0.0025923 | 0.132275 | 0.116407 | — | — |
| layered_global | 3 | 2 | 1 | 0.0024764 | 0.123748 | 0.109544 | — | — |
| layered_global | 3 | 2 | 2 | 0.0025604 | 0.132882 | 0.097657 | — | — |
| layered_global | 8 | 2 | 0 | 0.0151422 | 0.264471 | 0.213513 | — | — |
| layered_global | 8 | 2 | 1 | 0.0155812 | 0.249959 | 0.183251 | — | — |
| layered_global | 8 | 2 | 2 | 0.0175314 | 0.26575 | 0.0955772 | — | — |
| layered_global | 16 | 2 | 0 | 0.0896238 | 0.227792 | 0.172954 | — | — |
| layered_global | 16 | 2 | 1 | 0.0895776 | 0.218944 | 0.162655 | — | — |
| layered_global | 16 | 2 | 2 | 0.0844022 | 0.235658 | 0.167536 | — | — |
| layered_local | 3 | 2 | 0 | 0.0026783 | 0.175903 | 0.143861 | — | — |
| layered_local | 3 | 2 | 1 | 0.0024194 | 0.184933 | 0.0675585 | — | — |
| layered_local | 3 | 2 | 2 | 0.0024542 | 0.0976155 | 0.0642103 | — | — |
| layered_local | 8 | 2 | 0 | 0.0103576 | 0.127322 | 0.10579 | — | — |
| layered_local | 8 | 2 | 1 | 0.0101647 | 0.128288 | 0.0928325 | — | — |
| layered_local | 8 | 2 | 2 | 0.0122282 | 0.130954 | 0.11235 | — | — |
| layered_local | 16 | 2 | 0 | 0.0412066 | 0.212544 | 0.164596 | — | — |
| layered_local | 16 | 2 | 1 | 0.0347853 | 0.206761 | 0.14784 | — | — |
| layered_local | 16 | 2 | 2 | 0.0388208 | 0.191111 | 0.147748 | — | — |
| lcomp | 2 | — | 0 | 8.24e-05 | 0.0188803 | 0.0097558 | — | — |
| lcomp | 16 | — | 0 | 0.00035465 | 0.0193955 | 0.0117039 | — | — |
| lcomp | 128 | — | 0 | 0.00754595 | 0.0303236 | 0.0257786 | — | — |
| pivot | 2 | — | 0 | 0.0001199 | 0.0282628 | 0.0087431 | — | — |
| pivot | 16 | — | 0 | 0.00083095 | 0.0232333 | 0.0117121 | — | — |
| pivot | 128 | — | 0 | 0.0636114 | 0.0551715 | 0.0708494 | — | — |
| pivot_boundary | 2 | — | 0 | 0.0001521 | 0.0137022 | 0.0062421 | — | — |
| pivot_boundary | 16 | — | 0 | 0.0006724 | 0.0145259 | 0.0080925 | — | — |
| pivot_boundary | 128 | — | 0 | 0.0291256 | 0.0430272 | 0.0385787 | — | — |
| pivot_gadget | 2 | — | 0 | 0.00015635 | 0.0133553 | 0.0063602 | — | — |
| pivot_gadget | 16 | — | 0 | 0.00109875 | 0.0136858 | 0.0078862 | — | — |
| pivot_gadget | 128 | — | 0 | 0.335818 | 0.0407001 | 0.0410313 | — | — |
| random_clifford | 3 | 30 | 0 | 0.0024516 | 0.192679 | 0.164118 | — | — |
| random_clifford | 3 | 30 | 1 | 0.00300985 | 0.231956 | 0.129674 | — | — |
| random_clifford | 3 | 30 | 2 | 0.0030086 | 0.19833 | 0.171394 | — | — |
| random_clifford | 8 | 80 | 0 | 0.0171973 | 0.289646 | 0.213058 | — | — |
| random_clifford | 8 | 80 | 1 | 0.0194881 | 0.301126 | 0.231686 | — | — |
| random_clifford | 8 | 80 | 2 | 0.0172354 | 0.335308 | 0.130633 | — | — |
| random_clifford | 16 | 160 | 0 | 0.0910017 | 0.230858 | 0.141285 | — | — |
| random_clifford | 16 | 160 | 1 | 0.0861555 | 0.302881 | 0.217985 | — | — |
| random_clifford | 16 | 160 | 2 | 0.0904539 | 0.285667 | 0.18261 | — | — |
| random_clifford | 32 | 640 | 0 | 1.81422 | 1.00182 | 0.587372 | +4.5% | +0.8% |
| random_clifford | 32 | 640 | 1 | 2.17117 | 1.15756 | 0.714099 | -7.9% | -8.2% |
| random_clifford | 32 | 640 | 2 | 1.73401 | 1.02785 | 0.626272 | -6.9% | +7.3% |
| random_clifford | 64 | 1280 | 0 | 17.3607 | 2.81778 | 2.17587 | -1.1% | -8.7% |
| random_clifford | 64 | 1280 | 1 | 18.2016 | 3.06346 | 2.32331 | +7.9% | -7.8% |
| random_clifford | 64 | 1280 | 2 | 15.6657 | 2.97929 | 2.44828 | -5.7% | +2.5% |
| random_clifford | 128 | 2560 | 0 | 141.306 | 17.1721 | 16.9477 | -1.1% | -0.4% |
| random_clifford_t | 3 | 30 | 0 | 0.00243865 | 0.24058 | 0.159269 | — | — |
| random_clifford_t | 3 | 30 | 1 | 0.00264865 | 0.181097 | 0.167355 | — | — |
| random_clifford_t | 3 | 30 | 2 | 0.0021323 | 0.183534 | 0.126909 | — | — |
| random_clifford_t | 8 | 80 | 0 | 0.00944375 | 0.352905 | 0.248194 | — | — |
| random_clifford_t | 8 | 80 | 1 | 0.0118342 | 0.301165 | 0.203067 | — | — |
| random_clifford_t | 8 | 80 | 2 | 0.0117608 | 0.376521 | 0.273977 | — | — |
| random_clifford_t | 16 | 160 | 0 | 0.025789 | 0.386199 | 0.259036 | — | — |
| random_clifford_t | 16 | 160 | 1 | 0.0328254 | 0.285728 | 0.172987 | — | — |
| random_clifford_t | 16 | 160 | 2 | 0.0292903 | 0.268217 | 0.193394 | — | — |
| random_t_heavy | 3 | 30 | 0 | 0.0022573 | 0.160702 | 0.113182 | — | — |
| random_t_heavy | 3 | 30 | 1 | 0.0019727 | 0.143748 | 0.178816 | — | — |
| random_t_heavy | 3 | 30 | 2 | 0.00181585 | 0.17206 | 0.160959 | — | — |
| random_t_heavy | 8 | 80 | 0 | 0.0074576 | 0.220649 | 0.172758 | — | — |
| random_t_heavy | 8 | 80 | 1 | 0.00605315 | 0.222158 | 0.207642 | — | — |
| random_t_heavy | 8 | 80 | 2 | 0.00663435 | 0.232308 | 0.203389 | — | — |
| random_t_heavy | 16 | 160 | 0 | 0.0128312 | 0.324657 | 0.265184 | — | — |
| random_t_heavy | 16 | 160 | 1 | 0.0219374 | 0.347152 | 0.264222 | — | — |
| random_t_heavy | 16 | 160 | 2 | 0.0239288 | 0.434314 | 0.313818 | — | — |
| spider_fusion | 2 | — | 0 | 0.0001101 | 0.0440545 | 0.0096626 | — | — |
| spider_fusion | 16 | — | 0 | 0.00064485 | 0.0113444 | 0.0108314 | — | — |
| spider_fusion | 128 | — | 0 | 0.01562 | 0.0157603 | 0.0187321 | — | — |
| supplementarity | 2 | — | 0 | 8.925e-05 | 0.0113082 | 0.0027504 | — | — |
| supplementarity | 16 | — | 0 | 0.0001924 | 0.0123015 | 0.0033928 | — | — |
| supplementarity | 128 | — | 0 | 0.00180965 | 0.0326114 | 0.0204233 | — | — |

## Interpretation and evidence

The before/after percentages are observed timing differences, not causal guarantees or statistical significance tests. Standard current-code timings precede the pre-fix rerun; large configurations alternate all paths in a deterministic shuffled order. Frequency scaling, thermal state and matching order can affect results. Use the per-cell ranges and graph metrics in the raw records when interpreting changes.

[Full timings, including total time, ranges and verification status](timings.csv) · [Machine-readable timings](timings.json) · [Total-time table](TOTAL_TIMES.md) · [Historical comparisons with matching input hashes](historical-comparison.json) · [Standard run](current-standard/REPORT.md) · [Extended manifest and source hashes](comparison/environment.json)

Historical comparisons use the same workload specification and exact input hash. They can involve different source revisions or run conditions, so the contemporaneous pre-fix rerun is the stronger comparison for the audit changes. Neither comparison reruns old broken code on known invalid audit counterexamples.
