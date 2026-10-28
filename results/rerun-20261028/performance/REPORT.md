# Three-engine benchmark rerun — 2026-10-28

Status: COMPLETE. 160 planned workload configurations.

PyZX 0.10.3 and Memgraph 3.14.0. One warmup and three measured repetitions per engine/configuration. Each repetition uses the same input and deterministically shuffled engine order. Stock PyZX is timed once and shared by the two database comparisons. Cypher uses the optimized Python-coordinated Cypher schedule; hybrid additionally uses scoped C++ procedures. All verification and graph snapshot I/O are outside timers. Rewrite excludes graph import/export; total includes them (PyZX includes graph copying).

This is finite-size empirical scaling evidence, not a proof of asymptotic complexity. Random circuits use depth=20×qubits. Layered circuits use two layers with qubits−1 selected edges per layer. Rule fans and disjoint copies are different workload shapes. Local-layered seeds generate identical inputs; their repetitions are not independent circuits. QFT and other adversarial circuits are included in the separate correctness suite, not invented as new scalability families.

## Correctness against PyZX

- pyzx: `{"input_tensor": {"verified": 132, "inconclusive": 348}}`
- cypher: `{"input_tensor": {"verified": 132, "inconclusive": 348}, "tensor": {"verified": 141, "inconclusive": 339}, "graph": {"equal": 171, "inconclusive": 48, "different": 261}, "degree_sequence": {"equal": 222, "different": 258}}`
- hybrid: `{"input_tensor": {"verified": 132, "inconclusive": 348}, "tensor": {"verified": 141, "inconclusive": 339}, "graph": {"equal": 176, "inconclusive": 48, "different": 256}, "degree_sequence": {"equal": 224, "different": 256}}`

Tensor checks preserve boundary order and compare up to a nonzero global scalar. Labeled multigraph isomorphism preserves vertex type, phase, wire type/multiplicity and boundary positions, but excludes scalar. Isomorphism is bounded at 2,048 vertices, 4,096 edges and two seconds. Degree equality is only a weak structural check, not semantic proof. Different outputs or degree sequences can be equivalent because rewrite normal forms are not unique. Large tensor/isomorphism limits remain inconclusive. No global correctness claim is made for those outputs.

## Largest scale per family

Times in seconds; geometric mean of per-seed medians for circuit families. Speedup = PyZX/database; >1 favors the database. Failed, incomplete and no-op cells receive no speedup.

| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher speedup | Hybrid speedup |
|---|---:|---:|---:|---:|---:|---:|
| bialgebra | 128 | 2.6143 | 1.303 | 1.3959 | 2.0064 | 1.8729 |
| copies_bialgebra | 512 | 0.12294 | 13.313 | 12.914 | 0.0092346 | 0.0095205 |
| copies_copy | 512 | 0.02717 | 2.1113 | 2.1725 | 0.012869 | 0.012506 |
| copies_gadget_fusion | 512 | 0.019728 | 0.051985 | 0.050145 | 0.3795 | 0.39343 |
| copies_hadamard_cancel | 512 | 0.45208 | 15.545 | 15.808 | 0.029082 | 0.028598 |
| copies_identity | 512 | 0.44973 | 0.051324 | 0.0495 | 8.7626 | 9.0856 |
| copies_lcomp | 512 | 0.18542 | 0.14003 | 0.021001 | 1.3242 | 8.8289 |
| copies_pivot | 512 | 0.38773 | 0.26864 | 0.065943 | 1.4433 | 5.8797 |
| copies_pivot_boundary | 512 | 0.048272 | 0.097359 | 0.067842 | 0.49581 | 0.71153 |
| copies_pivot_gadget | 512 | 0.17519 | 0.20496 | 0.05475 | 0.85478 | 3.1999 |
| copies_spider_fusion | 512 | 0.5938 | 0.060628 | 0.068709 | 9.7942 | 8.6423 |
| copies_supplementarity | 512 | 0.024291 | 88.864 | 0.52323 | 0.00027336 | 0.046426 |
| copy | 128 | 0.0009342 | 0.010739 | 0.01131 | 0.086994 | 0.082599 |
| gadget_fusion | 128 | 0.0006133 | 0.040948 | 0.040663 | 0.014978 | 0.015083 |
| hadamard_cancel | 128 | 0.0081289 | 0.59069 | 0.59025 | 0.013762 | 0.013772 |
| identity | 128 | 0.0080776 | 0.011868 | 0.0116 | 0.68062 | 0.69634 |
| identity_chain | 512 | 0.017586 | 0.040604 | 0.050063 | 0.43311 | 0.35128 |
| layered_global | 128 | 141.86 | 1.4861 | 1.4861 | 95.459 | 95.459 |
| layered_local | 128 | 6.4711 | 4.0404 | 5.5544 | 1.6016 | 1.165 |
| lcomp | 128 | 0.0079456 | 0.0234 | 0.0198 | 0.33955 | 0.40129 |
| pivot | 128 | 0.03807 | 0.077784 | 0.078246 | 0.48943 | 0.48654 |
| pivot_boundary | 128 | 0.014673 | 0.039236 | 0.037284 | 0.37398 | 0.39355 |
| pivot_gadget | 128 | 0.25628 | 0.05136 | 0.043759 | 4.9898 | 5.8565 |
| random_clifford | 128 | 139.94 | 16.206 | 17.142 | 8.6348 | 8.1631 |
| random_clifford_t | 128 | 86.534 | 7.1084 | 7.5117 | 12.174 | 11.52 |
| random_t_heavy | 128 | 1.806 | 1.6243 | 1.1665 | 1.1119 | 1.5483 |
| spider_fusion | 128 | 0.01211 | 0.040563 | 0.039311 | 0.29854 | 0.30805 |
| supplementarity | 128 | 0.0012931 | 0.03283 | 0.021886 | 0.039388 | 0.059082 |

## Largest-scale total time

| Family | Size | PyZX total | Cypher total | Hybrid total | Cypher speedup | Hybrid speedup |
|---|---:|---:|---:|---:|---:|---:|
| bialgebra | 128 | 2.6159 | 1.9386 | 1.933 | 1.3494 | 1.3533 |
| copies_bialgebra | 512 | 0.13802 | 14.005 | 13.613 | 0.0098548 | 0.010139 |
| copies_copy | 512 | 0.040723 | 2.341 | 2.3954 | 0.017395 | 0.017 |
| copies_gadget_fusion | 512 | 0.034922 | 0.46853 | 0.4632 | 0.074535 | 0.075393 |
| copies_hadamard_cancel | 512 | 0.45975 | 15.911 | 16.069 | 0.028895 | 0.028611 |
| copies_identity | 512 | 0.45711 | 0.38301 | 0.41865 | 1.1934 | 1.0919 |
| copies_lcomp | 512 | 0.19746 | 0.39212 | 0.27272 | 0.50357 | 0.72405 |
| copies_pivot | 512 | 0.41272 | 0.89652 | 0.62369 | 0.46036 | 0.66173 |
| copies_pivot_boundary | 512 | 0.075577 | 0.90045 | 0.86701 | 0.083932 | 0.08717 |
| copies_pivot_gadget | 512 | 0.20056 | 0.85603 | 0.68183 | 0.2343 | 0.29415 |
| copies_spider_fusion | 512 | 0.60986 | 0.50725 | 0.53743 | 1.2023 | 1.1348 |
| copies_supplementarity | 512 | 0.04459 | 89.027 | 0.68711 | 0.00050086 | 0.064896 |
| copy | 128 | 0.0017185 | 0.047071 | 0.054642 | 0.036509 | 0.03145 |
| gadget_fusion | 128 | 0.0019491 | 0.0873 | 0.084616 | 0.022326 | 0.023035 |
| hadamard_cancel | 128 | 0.0092008 | 0.62908 | 0.62869 | 0.014626 | 0.014635 |
| identity | 128 | 0.0090227 | 0.060169 | 0.06289 | 0.14996 | 0.14347 |
| identity_chain | 512 | 0.019039 | 0.071896 | 0.082835 | 0.26481 | 0.22985 |
| layered_global | 128 | 141.87 | 1.8546 | 1.9014 | 76.498 | 74.616 |
| layered_local | 128 | 6.4792 | 5.1269 | 6.9068 | 1.2638 | 0.93809 |
| lcomp | 128 | 0.0093941 | 0.29519 | 0.29356 | 0.031824 | 0.032001 |
| pivot | 128 | 0.04106 | 0.69684 | 0.62761 | 0.058923 | 0.065422 |
| pivot_boundary | 128 | 0.01693 | 0.59381 | 0.57595 | 0.028511 | 0.029395 |
| pivot_gadget | 128 | 0.25843 | 0.60594 | 0.59121 | 0.4265 | 0.43712 |
| random_clifford | 128 | 139.95 | 17.064 | 18.019 | 8.2016 | 7.7672 |
| random_clifford_t | 128 | 86.551 | 8.4262 | 8.783 | 10.272 | 9.8544 |
| random_t_heavy | 128 | 1.8216 | 2.027 | 1.526 | 0.89866 | 1.1937 |
| spider_fusion | 128 | 0.01431 | 0.10523 | 0.10281 | 0.13599 | 0.1392 |
| supplementarity | 128 | 0.0023201 | 0.059816 | 0.051614 | 0.038787 | 0.044951 |

## Observed growth over the two largest sizes

Exponent = log(time ratio)/log(size ratio), based on rewrite medians aggregated across seeds. These local slopes are sensitive to fixed overhead, matching order and graph density; they are not Big-O estimates.

| Family | Sizes | PyZX exponent | Cypher exponent | Hybrid exponent |
|---|---|---:|---:|---:|
| bialgebra | 16→128 | 2.8285 | 2.1145 | 2.1606 |
| copies_bialgebra | 32→512 | 1.1423 | 0.9816 | 0.97904 |
| copies_copy | 32→512 | 1.1985 | 1.2845 | 1.3079 |
| copies_gadget_fusion | 32→512 | 1.343 | 0.33202 | 0.29231 |
| copies_hadamard_cancel | 32→512 | 1.8965 | 1.4886 | 1.5026 |
| copies_identity | 32→512 | 1.9006 | 0.21137 | 0.16352 |
| copies_lcomp | 32→512 | 1.6671 | 1.077 | 0.56124 |
| copies_pivot | 32→512 | 1.7156 | 0.85131 | 0.62484 |
| copies_pivot_boundary | 32→512 | 1.0728 | 0.40703 | 0.47172 |
| copies_pivot_gadget | 32→512 | 1.5248 | 1.1283 | 0.81342 |
| copies_spider_fusion | 32→512 | 1.8321 | 0.59018 | 0.63876 |
| copies_supplementarity | 32→512 | 1.1945 | 2.1678 | 1.5549 |
| copy | 16→128 | 0.75639 | 0.19238 | 0.22504 |
| gadget_fusion | 16→128 | 0.619 | 0.027233 | 0.053519 |
| hadamard_cancel | 16→128 | 1.644 | 0.96634 | 0.98935 |
| identity | 16→128 | 1.6508 | -0.6134 | -0.57628 |
| identity_chain | 128→512 | 1.668 | 0.33217 | 0.49014 |
| layered_global | 64→128 | 5.0537 | 0.98544 | 1.0619 |
| layered_local | 64→128 | 2.9532 | 2.1004 | 2.3935 |
| lcomp | 16→128 | 1.5825 | 0.37283 | 0.6121 |
| pivot | 16→128 | 1.8363 | 0.096756 | 0.60813 |
| pivot_boundary | 16→128 | 1.598 | 0.51365 | 0.74986 |
| pivot_gadget | 16→128 | 2.6069 | 0.37804 | 0.49164 |
| random_clifford | 64→128 | 3.0004 | 2.5546 | 2.8809 |
| random_clifford_t | 64→128 | 3.5778 | 2.1608 | 2.2532 |
| random_t_heavy | 64→128 | 1.8728 | 0.59694 | 0.58281 |
| spider_fusion | 16→128 | 1.4855 | 0.12081 | 0.087556 |
| supplementarity | 16→128 | 1.0813 | 0.5377 | 0.85647 |

## All scales

| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | PyZX total | Cypher total | Hybrid total |
|---|---:|---:|---:|---:|---:|---:|---:|
| bialgebra | 2 | 0.000203 | 0.010809 | 0.01164 | 0.0002896 | 0.023117 | 0.022286 |
| bialgebra | 16 | 0.0072933 | 0.016045 | 0.015619 | 0.0075165 | 0.042932 | 0.044527 |
| bialgebra | 128 | 2.6143 | 1.303 | 1.3959 | 2.6159 | 1.9386 | 1.933 |
| copies_bialgebra | 2 | 0.0003426 | 0.065868 | 0.07154 | 0.0004567 | 0.093349 | 0.10088 |
| copies_bialgebra | 32 | 0.0051795 | 0.87563 | 0.85538 | 0.0061532 | 0.94214 | 0.92157 |
| copies_bialgebra | 512 | 0.12294 | 13.313 | 12.914 | 0.13802 | 14.005 | 13.613 |
| copies_copy | 2 | 9.89e-05 | 0.0080399 | 0.0091671 | 0.0002048 | 0.018066 | 0.019598 |
| copies_copy | 32 | 0.0009795 | 0.059967 | 0.057821 | 0.001722 | 0.082139 | 0.080425 |
| copies_copy | 512 | 0.02717 | 2.1113 | 2.1725 | 0.040723 | 2.341 | 2.3954 |
| copies_gadget_fusion | 2 | 6.61e-05 | 0.018935 | 0.020791 | 0.0001833 | 0.03541 | 0.038749 |
| copies_gadget_fusion | 32 | 0.0004764 | 0.020706 | 0.022297 | 0.0014248 | 0.060357 | 0.061896 |
| copies_gadget_fusion | 512 | 0.019728 | 0.051985 | 0.050145 | 0.034922 | 0.46853 | 0.4632 |
| copies_hadamard_cancel | 2 | 6.79e-05 | 0.018502 | 0.018617 | 0.0001467 | 0.026941 | 0.02733 |
| copies_hadamard_cancel | 32 | 0.0023529 | 0.25072 | 0.24521 | 0.0028631 | 0.27458 | 0.26835 |
| copies_hadamard_cancel | 512 | 0.45208 | 15.545 | 15.808 | 0.45975 | 15.911 | 16.069 |
| copies_identity | 2 | 7.31e-05 | 0.030586 | 0.030975 | 0.0001764 | 0.053828 | 0.05341 |
| copies_identity | 32 | 0.0023145 | 0.028563 | 0.031456 | 0.0028412 | 0.063832 | 0.067378 |
| copies_identity | 512 | 0.44973 | 0.051324 | 0.0495 | 0.45711 | 0.38301 | 0.41865 |
| copies_lcomp | 2 | 0.0001235 | 0.0047371 | 0.0025202 | 0.0002185 | 0.012869 | 0.010329 |
| copies_lcomp | 32 | 0.0018229 | 0.0070695 | 0.0044304 | 0.002561 | 0.030488 | 0.028216 |
| copies_lcomp | 512 | 0.18542 | 0.14003 | 0.021001 | 0.19746 | 0.39212 | 0.27272 |
| copies_pivot | 2 | 0.0001702 | 0.022096 | 0.0089997 | 0.0003103 | 0.040554 | 0.026764 |
| copies_pivot | 32 | 0.0033321 | 0.025357 | 0.011662 | 0.0046253 | 0.072689 | 0.058981 |
| copies_pivot | 512 | 0.38773 | 0.26864 | 0.065943 | 0.41272 | 0.89652 | 0.62369 |
| copies_pivot_boundary | 2 | 0.0002216 | 0.024457 | 0.012917 | 0.000394 | 0.054295 | 0.043896 |
| copies_pivot_boundary | 32 | 0.0024656 | 0.031497 | 0.018344 | 0.0040438 | 0.10377 | 0.092967 |
| copies_pivot_boundary | 512 | 0.048272 | 0.097359 | 0.067842 | 0.075577 | 0.90045 | 0.86701 |
| copies_pivot_gadget | 2 | 0.0001944 | 0.026599 | 0.014522 | 0.0003407 | 0.057293 | 0.046589 |
| copies_pivot_gadget | 32 | 0.0025553 | 0.0089761 | 0.0057402 | 0.0040116 | 0.052916 | 0.050289 |
| copies_pivot_gadget | 512 | 0.17519 | 0.20496 | 0.05475 | 0.20056 | 0.85603 | 0.68183 |
| copies_spider_fusion | 2 | 0.0001479 | 0.0099183 | 0.0099361 | 0.0002925 | 0.019487 | 0.018951 |
| copies_spider_fusion | 32 | 0.0036946 | 0.011804 | 0.011691 | 0.0046822 | 0.041783 | 0.040993 |
| copies_spider_fusion | 512 | 0.5938 | 0.060628 | 0.068709 | 0.60986 | 0.50725 | 0.53743 |
| copies_supplementarity | 2 | 9.74e-05 | 0.015691 | 0.0032605 | 0.0002064 | 0.024345 | 0.011994 |
| copies_supplementarity | 32 | 0.0008855 | 0.21796 | 0.0070221 | 0.0016361 | 0.23602 | 0.024941 |
| copies_supplementarity | 512 | 0.024291 | 88.864 | 0.52323 | 0.04459 | 89.027 | 0.68711 |
| copy | 2 | 8.07e-05 | 0.0068037 | 0.0066548 | 0.0001683 | 0.016775 | 0.017007 |
| copy | 16 | 0.0001938 | 0.007198 | 0.0070833 | 0.0003592 | 0.020211 | 0.021339 |
| copy | 128 | 0.0009342 | 0.010739 | 0.01131 | 0.0017185 | 0.047071 | 0.054642 |
| gadget_fusion | 2 | 6.31e-05 | 0.03762 | 0.036322 | 0.0001757 | 0.063278 | 0.06176 |
| gadget_fusion | 16 | 0.0001693 | 0.038693 | 0.03638 | 0.0004819 | 0.068646 | 0.064457 |
| gadget_fusion | 128 | 0.0006133 | 0.040948 | 0.040663 | 0.0019491 | 0.0873 | 0.084616 |
| hadamard_cancel | 2 | 4.64e-05 | 0.013001 | 0.013514 | 0.0001119 | 0.020336 | 0.02119 |
| hadamard_cancel | 16 | 0.0002663 | 0.07919 | 0.075434 | 0.000439 | 0.092358 | 0.089945 |
| hadamard_cancel | 128 | 0.0081289 | 0.59069 | 0.59025 | 0.0092008 | 0.62908 | 0.62869 |
| identity | 2 | 4.71e-05 | 0.037361 | 0.038274 | 0.0001138 | 0.062843 | 0.064851 |
| identity | 16 | 0.0002609 | 0.042495 | 0.038449 | 0.000442 | 0.074758 | 0.068788 |
| identity | 128 | 0.0080776 | 0.011868 | 0.0116 | 0.0090227 | 0.060169 | 0.06289 |
| identity_chain | 2 | 5.05e-05 | 0.010282 | 0.010065 | 0.000115 | 0.017575 | 0.017849 |
| identity_chain | 16 | 0.0001553 | 0.01846 | 0.017001 | 0.0002683 | 0.026324 | 0.02455 |
| identity_chain | 128 | 0.0017416 | 0.02562 | 0.025376 | 0.0021193 | 0.038689 | 0.038016 |
| identity_chain | 512 | 0.017586 | 0.040604 | 0.050063 | 0.019039 | 0.071896 | 0.082835 |
| layered_global | 3 | 0.0023222 | 0.11271 | 0.082053 | 0.0025243 | 0.12561 | 0.09343 |
| layered_global | 8 | 0.015267 | 0.18686 | 0.14 | 0.015804 | 0.21055 | 0.1641 |
| layered_global | 16 | 0.078453 | 0.24426 | 0.16897 | 0.079321 | 0.28899 | 0.21151 |
| layered_global | 32 | 0.51175 | 0.55434 | 0.37099 | 0.51429 | 0.65359 | 0.47346 |
| layered_global | 64 | 4.2713 | 0.75059 | 0.71182 | 4.2806 | 1.01 | 1.0004 |
| layered_global | 128 | 141.86 | 1.4861 | 1.4861 | 141.87 | 1.8546 | 1.9014 |
| layered_local | 3 | 0.0023156 | 0.16378 | 0.11266 | 0.0025516 | 0.17794 | 0.12743 |
| layered_local | 8 | 0.0093829 | 0.25745 | 0.17594 | 0.0098912 | 0.28299 | 0.20238 |
| layered_local | 16 | 0.035389 | 0.3538 | 0.22817 | 0.036301 | 0.40261 | 0.27826 |
| layered_local | 32 | 0.17318 | 0.31922 | 0.3105 | 0.17498 | 0.4208 | 0.42467 |
| layered_local | 64 | 0.83554 | 0.9422 | 1.0571 | 0.83918 | 1.2933 | 1.4301 |
| layered_local | 128 | 6.4711 | 4.0404 | 5.5544 | 6.4792 | 5.1269 | 6.9068 |
| lcomp | 2 | 7.08e-05 | 0.01136 | 0.0061845 | 0.0001538 | 0.026208 | 0.020461 |
| lcomp | 16 | 0.0002958 | 0.010778 | 0.0055449 | 0.0004981 | 0.02985 | 0.026189 |
| lcomp | 128 | 0.0079456 | 0.0234 | 0.0198 | 0.0093941 | 0.29519 | 0.29356 |
| pivot | 2 | 0.0001014 | 0.05208 | 0.017713 | 0.0002031 | 0.081687 | 0.047086 |
| pivot | 16 | 0.000836 | 0.063608 | 0.022093 | 0.0011741 | 0.11232 | 0.072505 |
| pivot | 128 | 0.03807 | 0.077784 | 0.078246 | 0.04106 | 0.69684 | 0.62761 |
| pivot_boundary | 2 | 0.0001197 | 0.009545 | 0.0047748 | 0.000238 | 0.022401 | 0.01792 |
| pivot_boundary | 16 | 0.0005289 | 0.013484 | 0.0078403 | 0.0008924 | 0.043534 | 0.037652 |
| pivot_boundary | 128 | 0.014673 | 0.039236 | 0.037284 | 0.01693 | 0.59381 | 0.57595 |
| pivot_gadget | 2 | 0.0001427 | 0.023221 | 0.010355 | 0.0002688 | 0.047348 | 0.034684 |
| pivot_gadget | 16 | 0.0011336 | 0.0234 | 0.015743 | 0.0014913 | 0.064684 | 0.060565 |
| pivot_gadget | 128 | 0.25628 | 0.05136 | 0.043759 | 0.25843 | 0.60594 | 0.59121 |
| random_clifford | 3 | 0.0063167 | 0.35678 | 0.22776 | 0.0067889 | 0.37967 | 0.24868 |
| random_clifford | 8 | 0.040451 | 0.46556 | 0.25731 | 0.041477 | 0.50163 | 0.29467 |
| random_clifford | 16 | 0.26645 | 0.45145 | 0.25761 | 0.26903 | 0.51544 | 0.31973 |
| random_clifford | 32 | 1.9739 | 1.037 | 0.6256 | 1.978 | 1.1648 | 0.75474 |
| random_clifford | 64 | 17.488 | 2.7585 | 2.3272 | 17.496 | 3.0636 | 2.6218 |
| random_clifford | 128 | 139.94 | 16.206 | 17.142 | 139.95 | 17.064 | 18.019 |
| random_clifford_t | 3 | 0.0054241 | 0.19895 | 0.15389 | 0.0060244 | 0.21529 | 0.17066 |
| random_clifford_t | 8 | 0.027708 | 0.31611 | 0.21118 | 0.028691 | 0.34666 | 0.24185 |
| random_clifford_t | 16 | 0.10932 | 0.57691 | 0.34549 | 0.11133 | 0.64289 | 0.43779 |
| random_clifford_t | 32 | 0.54235 | 0.81922 | 0.55842 | 0.54944 | 0.96198 | 0.70521 |
| random_clifford_t | 64 | 7.2472 | 1.5897 | 1.5756 | 7.2556 | 1.9998 | 1.9687 |
| random_clifford_t | 128 | 86.534 | 7.1084 | 7.5117 | 86.551 | 8.4262 | 8.783 |
| random_t_heavy | 3 | 0.0032637 | 0.20693 | 0.16216 | 0.003642 | 0.22209 | 0.17647 |
| random_t_heavy | 8 | 0.017536 | 0.32815 | 0.24368 | 0.018472 | 0.35651 | 0.27351 |
| random_t_heavy | 16 | 0.04598 | 0.32926 | 0.23868 | 0.047746 | 0.37508 | 0.28383 |
| random_t_heavy | 32 | 0.13824 | 0.7644 | 0.55141 | 0.14197 | 0.85475 | 0.64087 |
| random_t_heavy | 64 | 0.49312 | 1.0739 | 0.7788 | 0.50046 | 1.2605 | 0.94847 |
| random_t_heavy | 128 | 1.806 | 1.6243 | 1.1665 | 1.8216 | 2.027 | 1.526 |
| spider_fusion | 2 | 8.97e-05 | 0.035446 | 0.033216 | 0.0001891 | 0.054998 | 0.052938 |
| spider_fusion | 16 | 0.0005516 | 0.031553 | 0.032768 | 0.0008368 | 0.055702 | 0.056872 |
| spider_fusion | 128 | 0.01211 | 0.040563 | 0.039311 | 0.01431 | 0.10523 | 0.10281 |
| supplementarity | 2 | 8.54e-05 | 0.010367 | 0.0031035 | 0.0001616 | 0.017623 | 0.011299 |
| supplementarity | 16 | 0.0001365 | 0.010732 | 0.0036873 | 0.0003102 | 0.021069 | 0.013854 |
| supplementarity | 128 | 0.0012931 | 0.03283 | 0.021886 | 0.0023201 | 0.059816 | 0.051614 |

Individual seed/configuration medians: [timings.csv](timings.csv). All trials, checks and timings: [observations.jsonl](observations.jsonl). Exact source archive: [sources.zip](sources.zip). Regressions and all 223 correctness fixtures: [validation report](../validation/REPORT.md).

## Errors

