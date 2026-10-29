# Three-engine benchmark rerun

Status: PARTIAL. 257 planned workload configurations.

PyZX 0.10.3 and Memgraph 3.14.0. One warmup and three measured repetitions per engine/configuration. Each repetition uses the same input and deterministically shuffled engine order. Stock PyZX is timed once and shared by the two database comparisons. Cypher uses the optimized Python-coordinated Cypher schedule; hybrid additionally uses scoped C++ procedures. All verification and graph snapshot I/O are outside timers. Rewrite excludes graph import/export; total includes them (PyZX includes graph copying).

This is finite-size empirical scaling evidence, not a proof of asymptotic complexity. Random circuits use depth=20×qubits. Layered circuits use two layers with qubits−1 selected edges per layer. Rule fans and disjoint copies are different workload shapes. Local-layered seeds generate identical inputs; their repetitions are not independent circuits. QFT and other adversarial circuits are included in the separate correctness suite, not invented as new scalability families.

## Correctness against PyZX

- pyzx: `{"input_tensor": {"verified": 138, "inconclusive": 494}}`
- cypher: `{"input_tensor": {"verified": 138, "inconclusive": 494}, "tensor": {"verified": 156, "inconclusive": 476}, "graph": {"different": 272, "equal": 198, "inconclusive": 162}, "degree_sequence": {"different": 269, "equal": 363}}`
- hybrid: `{"input_tensor": {"verified": 138, "inconclusive": 494}, "tensor": {"verified": 156, "inconclusive": 476}, "graph": {"equal": 204, "different": 266, "inconclusive": 162}, "degree_sequence": {"equal": 366, "different": 266}}`

Tensor checks preserve boundary order and compare up to a nonzero global scalar. Labeled multigraph isomorphism preserves vertex type, phase, wire type/multiplicity and boundary positions, but excludes scalar. Isomorphism is bounded at 2,048 vertices, 4,096 edges and two seconds. Degree equality is only a weak structural check, not semantic proof. Different outputs or degree sequences can be equivalent because rewrite normal forms are not unique. Large tensor/isomorphism limits remain inconclusive. No global correctness claim is made for those outputs.

## Largest scale per family

Times in seconds; geometric mean of per-seed medians for circuit families. Speedup = PyZX/database; >1 favors the database. Failed, incomplete and no-op cells receive no speedup.

| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | Cypher speedup | Hybrid speedup |
|---|---:|---:|---:|---:|---:|---:|
| bialgebra | 1024 | — | — | — | — | — |
| copies_bialgebra | 16384 | — | — | — | — | — |
| copies_copy | 16384 | — | — | — | — | — |
| copies_gadget_fusion | 16384 | — | — | — | — | — |
| copies_hadamard_cancel | 16384 | — | — | — | — | — |
| copies_identity | 16384 | — | — | — | — | — |
| copies_lcomp | 16384 | — | — | — | — | — |
| copies_pivot | 16384 | — | — | — | — | — |
| copies_pivot_boundary | 16384 | — | — | — | — | — |
| copies_pivot_gadget | 16384 | — | — | — | — | — |
| copies_spider_fusion | 16384 | — | — | — | — | — |
| copies_supplementarity | 16384 | — | — | — | — | — |
| copy | 16384 | — | — | — | — | — |
| gadget_fusion | 16384 | — | — | — | — | — |
| hadamard_cancel | 16384 | — | — | — | — | — |
| identity | 16384 | — | — | — | — | — |
| identity_chain | 16384 | — | — | — | — | — |
| layered_global | 256 | — | — | — | — | — |
| layered_local | 256 | — | — | — | — | — |
| lcomp | 1024 | 0.63959 | 0.65662 | 0.7109 | 0.97407 | 0.89968 |
| pivot | 1024 | 3.4471 | 1.4591 | 3.3108 | 2.3624 | 1.0412 |
| pivot_boundary | 1024 | 1.9901 | 1.5305 | 1.5879 | 1.3002 | 1.2533 |
| pivot_gadget | 1024 | 116.2 | 1.5254 | 1.6201 | 76.178 | 71.722 |
| random_clifford | 256 | — | — | — | — | — |
| random_clifford_t | 256 | — | — | — | — | — |
| random_t_heavy | 256 | — | — | — | — | — |
| spider_fusion | 16384 | — | — | — | — | — |
| supplementarity | 16384 | — | — | — | — | — |

## Largest-scale total time

| Family | Size | PyZX total | Cypher total | Hybrid total | Cypher speedup | Hybrid speedup |
|---|---:|---:|---:|---:|---:|---:|
| bialgebra | 1024 | — | — | — | — | — |
| copies_bialgebra | 16384 | — | — | — | — | — |
| copies_copy | 16384 | — | — | — | — | — |
| copies_gadget_fusion | 16384 | — | — | — | — | — |
| copies_hadamard_cancel | 16384 | — | — | — | — | — |
| copies_identity | 16384 | — | — | — | — | — |
| copies_lcomp | 16384 | — | — | — | — | — |
| copies_pivot | 16384 | — | — | — | — | — |
| copies_pivot_boundary | 16384 | — | — | — | — | — |
| copies_pivot_gadget | 16384 | — | — | — | — | — |
| copies_spider_fusion | 16384 | — | — | — | — | — |
| copies_supplementarity | 16384 | — | — | — | — | — |
| copy | 16384 | — | — | — | — | — |
| gadget_fusion | 16384 | — | — | — | — | — |
| hadamard_cancel | 16384 | — | — | — | — | — |
| identity | 16384 | — | — | — | — | — |
| identity_chain | 16384 | — | — | — | — | — |
| layered_global | 256 | — | — | — | — | — |
| layered_local | 256 | — | — | — | — | — |
| lcomp | 1024 | 0.65146 | 16.384 | 16.371 | 0.039763 | 0.039794 |
| pivot | 1024 | 3.4637 | 33.694 | 35.708 | 0.1028 | 0.097001 |
| pivot_boundary | 1024 | 2.0057 | 33.862 | 34.11 | 0.059233 | 0.058803 |
| pivot_gadget | 1024 | 116.22 | 33.612 | 33.19 | 3.4576 | 3.5015 |
| random_clifford | 256 | — | — | — | — | — |
| random_clifford_t | 256 | — | — | — | — | — |
| random_t_heavy | 256 | — | — | — | — | — |
| spider_fusion | 16384 | — | — | — | — | — |
| supplementarity | 16384 | — | — | — | — | — |

## Observed growth over the two largest sizes

Exponent = log(time ratio)/log(size ratio), based on rewrite medians aggregated across seeds. These local slopes are sensitive to fixed overhead, matching order and graph density; they are not Big-O estimates.

| Family | Sizes | PyZX exponent | Cypher exponent | Hybrid exponent |
|---|---|---:|---:|---:|
| bialgebra | 512→1024 | — | — | — |
| copies_bialgebra | 8192→16384 | — | — | — |
| copies_copy | 8192→16384 | — | — | — |
| copies_gadget_fusion | 8192→16384 | — | — | — |
| copies_hadamard_cancel | 8192→16384 | — | — | — |
| copies_identity | 8192→16384 | — | — | — |
| copies_lcomp | 8192→16384 | — | — | — |
| copies_pivot | 8192→16384 | — | — | — |
| copies_pivot_boundary | 8192→16384 | — | — | — |
| copies_pivot_gadget | 8192→16384 | — | — | — |
| copies_spider_fusion | 8192→16384 | — | — | — |
| copies_supplementarity | 8192→16384 | — | — | — |
| copy | 4096→16384 | — | — | — |
| gadget_fusion | 4096→16384 | — | — | — |
| hadamard_cancel | 4096→16384 | — | — | — |
| identity | 4096→16384 | — | — | — |
| identity_chain | 8192→16384 | — | — | — |
| layered_global | 128→256 | — | — | — |
| layered_local | 128→256 | — | — | — |
| lcomp | 512→1024 | 1.9867 | 1.9174 | 1.9091 |
| pivot | 512→1024 | 2.1155 | 1.9332 | 2.0081 |
| pivot_boundary | 512→1024 | 2.1097 | 1.9199 | 1.8429 |
| pivot_gadget | 512→1024 | 3.2092 | 2.0129 | 1.9472 |
| random_clifford | 128→256 | — | — | — |
| random_clifford_t | 128→256 | — | — | — |
| random_t_heavy | 128→256 | — | — | — |
| spider_fusion | 4096→16384 | — | — | — |
| supplementarity | 4096→16384 | — | — | — |

## All scales

| Family | Size | PyZX rewrite | Cypher rewrite | Hybrid rewrite | PyZX total | Cypher total | Hybrid total |
|---|---:|---:|---:|---:|---:|---:|---:|
| bialgebra | 2 | 0.0001893 | 0.0069492 | 0.0063317 | 0.0002648 | 0.01513 | 0.014389 |
| bialgebra | 16 | 0.007055 | 0.0092712 | 0.010139 | 0.0072791 | 0.031949 | 0.032664 |
| bialgebra | 128 | 2.6998 | 1.343 | 1.3113 | 2.7016 | 1.9284 | 1.9047 |
| bialgebra | 512 | — | — | — | — | — | — |
| bialgebra | 1024 | — | — | — | — | — | — |
| copies_bialgebra | 2 | 0.0003503 | 0.009831 | 0.0095001 | 0.0004558 | 0.019955 | 0.018634 |
| copies_bialgebra | 32 | 0.0050954 | 0.14853 | 0.14423 | 0.0059566 | 0.19603 | 0.1926 |
| copies_bialgebra | 512 | 0.15739 | 11.936 | 12.018 | 0.17837 | 12.643 | 12.921 |
| copies_bialgebra | 2048 | 1.0538 | 183.84 | 182.13 | 1.1236 | 186.59 | 184.95 |
| copies_bialgebra | 4096 | 3.4537 | 808.77 | 798.07 | 3.5606 | 814.68 | 804.03 |
| copies_bialgebra | 8192 | — | — | — | — | — | — |
| copies_bialgebra | 16384 | — | — | — | — | — | — |
| copies_copy | 2 | 0.0001013 | 0.0068447 | 0.0064706 | 0.0002072 | 0.015548 | 0.014534 |
| copies_copy | 32 | 0.0010065 | 0.049503 | 0.049351 | 0.0017716 | 0.069551 | 0.069283 |
| copies_copy | 512 | 0.026681 | 2.0448 | 2.1687 | 0.037252 | 2.2643 | 2.3893 |
| copies_copy | 2048 | 0.25366 | 20.3 | 20.805 | 0.30212 | 21.15 | 21.677 |
| copies_copy | 4096 | 0.89994 | 77.716 | 79.489 | 0.9895 | 79.429 | 81.132 |
| copies_copy | 8192 | — | — | — | — | — | — |
| copies_copy | 16384 | — | — | — | — | — | — |
| copies_gadget_fusion | 2 | 6.88e-05 | 0.0072906 | 0.0077587 | 0.0001913 | 0.016487 | 0.016272 |
| copies_gadget_fusion | 32 | 0.0004775 | 0.0081105 | 0.0079785 | 0.0013741 | 0.039613 | 0.039128 |
| copies_gadget_fusion | 512 | 0.018436 | 0.03385 | 0.036654 | 0.032421 | 0.49273 | 0.46255 |
| copies_gadget_fusion | 2048 | 0.25397 | 0.13176 | 0.13636 | 0.31668 | 1.8109 | 1.8172 |
| copies_gadget_fusion | 4096 | 1.0598 | 0.2455 | 0.24514 | 1.1823 | 3.4094 | 3.4051 |
| copies_gadget_fusion | 8192 | 4.0681 | 0.42657 | 0.43474 | 4.3006 | 7.5206 | 7.8853 |
| copies_gadget_fusion | 16384 | — | — | — | — | — | — |
| copies_hadamard_cancel | 2 | 7.15e-05 | 0.018889 | 0.017755 | 0.0001521 | 0.027575 | 0.025756 |
| copies_hadamard_cancel | 32 | 0.0022758 | 0.24949 | 0.26371 | 0.003202 | 0.27347 | 0.28605 |
| copies_hadamard_cancel | 512 | 0.46438 | 15.312 | 15.28 | 0.47174 | 15.585 | 15.636 |
| copies_hadamard_cancel | 2048 | 7.2555 | 205.37 | 205.12 | 7.2892 | 206.51 | 206.11 |
| copies_hadamard_cancel | 4096 | — | — | — | — | — | — |
| copies_hadamard_cancel | 8192 | — | — | — | — | — | — |
| copies_hadamard_cancel | 16384 | — | — | — | — | — | — |
| copies_identity | 2 | 7.28e-05 | 0.0073693 | 0.0067417 | 0.0001581 | 0.016227 | 0.014466 |
| copies_identity | 32 | 0.0022628 | 0.0071834 | 0.0069922 | 0.0027552 | 0.028394 | 0.027864 |
| copies_identity | 512 | 0.47099 | 0.030731 | 0.024912 | 0.47836 | 0.31406 | 0.31029 |
| copies_identity | 2048 | 7.3443 | 0.077445 | 0.079691 | 7.3731 | 1.4144 | 1.4237 |
| copies_identity | 4096 | 28.885 | 0.14549 | 0.1638 | 28.945 | 2.507 | 2.3394 |
| copies_identity | 8192 | 121.06 | 0.29448 | 0.29384 | 121.18 | 5.1614 | 5.0431 |
| copies_identity | 16384 | — | — | — | — | — | — |
| copies_lcomp | 2 | 0.000116 | 0.0051669 | 0.0031321 | 0.0002237 | 0.01412 | 0.011761 |
| copies_lcomp | 32 | 0.0018253 | 0.006361 | 0.0036301 | 0.0025465 | 0.028599 | 0.024879 |
| copies_lcomp | 512 | 0.19175 | 0.13688 | 0.02218 | 0.20217 | 0.42734 | 0.27877 |
| copies_lcomp | 2048 | 2.8647 | 2.4903 | 0.065598 | 2.9182 | 3.5472 | 1.1 |
| copies_lcomp | 4096 | 11.162 | 20.786 | 0.13036 | 11.25 | 22.776 | 2.1533 |
| copies_lcomp | 8192 | 47.782 | 112.29 | 0.25192 | 47.963 | 116.8 | 4.6039 |
| copies_lcomp | 16384 | — | — | — | — | — | — |
| copies_pivot | 2 | 0.0001687 | 0.0095412 | 0.003967 | 0.0003145 | 0.019247 | 0.013802 |
| copies_pivot | 32 | 0.0032878 | 0.011842 | 0.0067457 | 0.0045087 | 0.051737 | 0.044511 |
| copies_pivot | 512 | 0.39255 | 0.19397 | 0.055037 | 0.41969 | 0.77391 | 0.65259 |
| copies_pivot | 2048 | 5.749 | 4.3075 | 0.20576 | 5.8313 | 6.5287 | 2.4375 |
| copies_pivot | 4096 | 23.331 | 30.088 | 0.40755 | 23.487 | 34.653 | 4.91 |
| copies_pivot | 8192 | — | — | — | — | — | — |
| copies_pivot | 16384 | — | — | — | — | — | — |
| copies_pivot_boundary | 2 | 0.0002403 | 0.0060501 | 0.0026983 | 0.000411 | 0.016984 | 0.013006 |
| copies_pivot_boundary | 32 | 0.0023079 | 0.0095824 | 0.0060647 | 0.0038112 | 0.061724 | 0.057303 |
| copies_pivot_boundary | 512 | 0.042978 | 0.066375 | 0.056008 | 0.066898 | 0.88599 | 0.83661 |
| copies_pivot_boundary | 2048 | 0.39836 | 0.28571 | 0.21286 | 0.51864 | 3.3536 | 3.2602 |
| copies_pivot_boundary | 4096 | 1.0993 | 0.57253 | 0.40328 | 1.3102 | 6.9206 | 6.8737 |
| copies_pivot_boundary | 8192 | — | — | — | — | — | — |
| copies_pivot_boundary | 16384 | — | — | — | — | — | — |
| copies_pivot_gadget | 2 | 0.0002011 | 0.0062684 | 0.0033459 | 0.0003964 | 0.017295 | 0.013142 |
| copies_pivot_gadget | 32 | 0.0024916 | 0.010634 | 0.0057304 | 0.0039651 | 0.068051 | 0.05637 |
| copies_pivot_gadget | 512 | 0.19901 | 0.19491 | 0.049626 | 0.22075 | 0.89905 | 0.69579 |
| copies_pivot_gadget | 2048 | 3.7421 | 4.5452 | 0.19719 | 3.8377 | 7.0477 | 3.0018 |
| copies_pivot_gadget | 4096 | 16.395 | 28.299 | 0.37251 | 16.646 | 34.326 | 5.7313 |
| copies_pivot_gadget | 8192 | — | — | — | — | — | — |
| copies_pivot_gadget | 16384 | — | — | — | — | — | — |
| copies_spider_fusion | 2 | 0.0001474 | 0.0094309 | 0.009493 | 0.0002679 | 0.018478 | 0.018565 |
| copies_spider_fusion | 32 | 0.0036308 | 0.011448 | 0.011892 | 0.0046242 | 0.042004 | 0.042087 |
| copies_spider_fusion | 512 | 0.63729 | 0.056061 | 0.054791 | 0.6546 | 0.46275 | 0.46028 |
| copies_spider_fusion | 2048 | 9.3882 | 0.18296 | 0.17037 | 9.4519 | 1.9691 | 1.9227 |
| copies_spider_fusion | 4096 | 39.773 | 0.31709 | 0.3329 | 39.956 | 3.6965 | 3.495 |
| copies_spider_fusion | 8192 | — | — | — | — | — | — |
| copies_spider_fusion | 16384 | — | — | — | — | — | — |
| copies_supplementarity | 2 | 0.0001043 | 0.013483 | 0.0025023 | 0.0002105 | 0.021305 | 0.0097911 |
| copies_supplementarity | 32 | 0.0009559 | 0.1829 | 0.0068017 | 0.0018573 | 0.19904 | 0.025496 |
| copies_supplementarity | 512 | 0.019336 | 88.819 | 0.43289 | 0.035579 | 88.97 | 0.58813 |
| copies_supplementarity | 2048 | — | — | — | — | — | — |
| copies_supplementarity | 4096 | — | — | — | — | — | — |
| copies_supplementarity | 8192 | — | — | — | — | — | — |
| copies_supplementarity | 16384 | — | — | — | — | — | — |
| copy | 2 | 6.64e-05 | 0.0047557 | 0.0045337 | 0.0001448 | 0.012655 | 0.011456 |
| copy | 16 | 0.0001696 | 0.0046216 | 0.0041262 | 0.000341 | 0.014951 | 0.013891 |
| copy | 128 | 0.000929 | 0.0055995 | 0.0050636 | 0.0017667 | 0.034056 | 0.033168 |
| copy | 512 | 0.003696 | 0.0092208 | 0.011066 | 0.0065664 | 0.1064 | 0.1023 |
| copy | 1024 | 0.012609 | 0.0155 | 0.012784 | 0.019601 | 0.24634 | 0.25929 |
| copy | 4096 | 0.029872 | 0.039514 | 0.042882 | 0.052087 | 0.80235 | 0.72098 |
| copy | 16384 | — | — | — | — | — | — |
| gadget_fusion | 2 | 5.16e-05 | 0.0079266 | 0.0074759 | 0.0001427 | 0.016611 | 0.016247 |
| gadget_fusion | 16 | 0.0001281 | 0.0074326 | 0.0070024 | 0.0003953 | 0.01731 | 0.018159 |
| gadget_fusion | 128 | 0.0005928 | 0.0096784 | 0.0092036 | 0.0018113 | 0.033525 | 0.03104 |
| gadget_fusion | 512 | 0.0023189 | 0.017182 | 0.017472 | 0.0076674 | 0.090197 | 0.085249 |
| gadget_fusion | 1024 | 0.0046913 | 0.029988 | 0.02817 | 0.014028 | 0.15942 | 0.16359 |
| gadget_fusion | 4096 | 0.017878 | 0.13943 | 0.12711 | 0.05746 | 0.65804 | 0.59138 |
| gadget_fusion | 16384 | — | — | — | — | — | — |
| hadamard_cancel | 2 | 7.79e-05 | 0.013412 | 0.012904 | 0.0001742 | 0.021994 | 0.020023 |
| hadamard_cancel | 16 | 0.000262 | 0.05305 | 0.05282 | 0.0004411 | 0.064095 | 0.063647 |
| hadamard_cancel | 128 | 0.0085149 | 0.59901 | 0.57842 | 0.010056 | 0.63812 | 0.62038 |
| hadamard_cancel | 512 | 0.12522 | 4.602 | 4.5332 | 0.12896 | 4.7362 | 4.6847 |
| hadamard_cancel | 1024 | 0.45806 | 15.42 | 15.259 | 0.46549 | 15.718 | 15.53 |
| hadamard_cancel | 4096 | 7.8473 | 208.65 | 209.74 | 7.8759 | 209.75 | 210.99 |
| hadamard_cancel | 16384 | — | — | — | — | — | — |
| identity | 2 | 4.68e-05 | 0.007265 | 0.0067317 | 0.0001115 | 0.015361 | 0.013882 |
| identity | 16 | 0.0002623 | 0.0064441 | 0.0065784 | 0.0004618 | 0.016736 | 0.01686 |
| identity | 128 | 0.0080104 | 0.008165 | 0.0086505 | 0.0089778 | 0.045252 | 0.047052 |
| identity | 512 | 0.13508 | 0.017358 | 0.018629 | 0.13905 | 0.18494 | 0.17499 |
| identity | 1024 | 0.48667 | 0.023327 | 0.02795 | 0.49407 | 0.28653 | 0.30523 |
| identity | 4096 | 9.3226 | 0.083595 | 0.097898 | 9.361 | 1.6211 | 1.7292 |
| identity | 16384 | — | — | — | — | — | — |
| identity_chain | 2 | 4.5e-05 | 0.010853 | 0.0090245 | 0.0001065 | 0.01804 | 0.016163 |
| identity_chain | 16 | 0.0001648 | 0.019767 | 0.019773 | 0.0002762 | 0.028171 | 0.028543 |
| identity_chain | 128 | 0.001957 | 0.028803 | 0.027504 | 0.0026379 | 0.042846 | 0.041343 |
| identity_chain | 512 | 0.017164 | 0.036568 | 0.036711 | 0.01858 | 0.067814 | 0.067426 |
| identity_chain | 2048 | 0.25515 | 0.073115 | 0.079824 | 0.26147 | 0.17955 | 0.18402 |
| identity_chain | 4096 | 1.3667 | 0.17646 | 0.17597 | 1.3898 | 0.42879 | 0.44952 |
| identity_chain | 8192 | — | — | — | — | — | — |
| identity_chain | 16384 | — | — | — | — | — | — |
| layered_global | 3 | 0.0030328 | 0.10922 | 0.088466 | 0.0033225 | 0.12298 | 0.10177 |
| layered_global | 8 | 0.015022 | 0.15426 | 0.10857 | 0.015532 | 0.17488 | 0.13024 |
| layered_global | 16 | 0.074854 | 0.18877 | 0.1375 | 0.075687 | 0.22999 | 0.17828 |
| layered_global | 32 | 0.5006 | 0.30715 | 0.26324 | 0.50297 | 0.39578 | 0.36047 |
| layered_global | 64 | 4.2377 | 0.77694 | 0.74319 | 4.2461 | 1.0444 | 1.0148 |
| layered_global | 128 | 137.35 | 1.4532 | 1.3975 | 137.36 | 1.8265 | 1.7562 |
| layered_global | 256 | — | — | — | — | — | — |
| layered_local | 3 | 0.0034519 | 0.10192 | 0.076343 | 0.0037828 | 0.11492 | 0.088582 |
| layered_local | 8 | 0.0091563 | 0.12484 | 0.091455 | 0.0096212 | 0.1448 | 0.11192 |
| layered_local | 16 | 0.034685 | 0.15377 | 0.12733 | 0.035557 | 0.19236 | 0.16742 |
| layered_local | 32 | 0.1688 | 0.26137 | 0.265 | 0.17055 | 0.3682 | 0.37483 |
| layered_local | 64 | 0.78432 | 0.80221 | 0.90902 | 0.78777 | 1.1306 | 1.2663 |
| layered_local | 128 | 6.075 | 3.9571 | 5.3419 | 6.082 | 5.0642 | 6.6352 |
| layered_local | 256 | — | — | — | — | — | — |
| lcomp | 2 | 0.0001129 | 0.0050321 | 0.0024918 | 0.0001993 | 0.012844 | 0.0098229 |
| lcomp | 16 | 0.0002883 | 0.005166 | 0.0028664 | 0.0004813 | 0.019075 | 0.017219 |
| lcomp | 128 | 0.0068899 | 0.018567 | 0.016374 | 0.0080311 | 0.289 | 0.29064 |
| lcomp | 512 | 0.16138 | 0.17382 | 0.18929 | 0.16572 | 4.1377 | 4.3736 |
| lcomp | 1024 | 0.63959 | 0.65662 | 0.7109 | 0.65146 | 16.384 | 16.371 |
| pivot | 2 | 0.0001198 | 0.0086058 | 0.0039454 | 0.0002175 | 0.017785 | 0.011874 |
| pivot | 16 | 0.0010102 | 0.0083821 | 0.006584 | 0.001354 | 0.029779 | 0.031068 |
| pivot | 128 | 0.040629 | 0.036835 | 0.063649 | 0.043347 | 0.58977 | 0.6749 |
| pivot | 512 | 0.79546 | 0.38209 | 0.82305 | 0.80415 | 8.5719 | 9.0155 |
| pivot | 1024 | 3.4471 | 1.4591 | 3.3108 | 3.4637 | 33.694 | 35.708 |
| pivot_boundary | 2 | 0.0001199 | 0.005969 | 0.0026866 | 0.0002332 | 0.015396 | 0.011399 |
| pivot_boundary | 16 | 0.0007684 | 0.0065297 | 0.003931 | 0.0013929 | 0.036405 | 0.033231 |
| pivot_boundary | 128 | 0.014244 | 0.033199 | 0.032635 | 0.016331 | 0.65427 | 0.57754 |
| pivot_boundary | 512 | 0.46108 | 0.40448 | 0.44266 | 0.46932 | 8.4227 | 8.5699 |
| pivot_boundary | 1024 | 1.9901 | 1.5305 | 1.5879 | 2.0057 | 33.862 | 34.11 |
| pivot_gadget | 2 | 0.0001364 | 0.0059564 | 0.002983 | 0.0002476 | 0.014771 | 0.011622 |
| pivot_gadget | 16 | 0.0010519 | 0.0059407 | 0.0031251 | 0.0014349 | 0.026746 | 0.023325 |
| pivot_gadget | 128 | 0.20992 | 0.032956 | 0.033617 | 0.21215 | 0.60046 | 0.61532 |
| pivot_gadget | 512 | 12.564 | 0.37794 | 0.42012 | 12.573 | 8.6739 | 8.585 |
| pivot_gadget | 1024 | 116.2 | 1.5254 | 1.6201 | 116.22 | 33.612 | 33.19 |
| random_clifford | 3 | 0.0081038 | 0.19484 | 0.14909 | 0.0085909 | 0.21465 | 0.16932 |
| random_clifford | 8 | 0.036272 | 0.22375 | 0.13023 | 0.037269 | 0.25313 | 0.15668 |
| random_clifford | 16 | 0.23406 | 0.33956 | 0.19992 | 0.23588 | 0.38985 | 0.24929 |
| random_clifford | 32 | 1.596 | 0.79897 | 0.50415 | 1.5996 | 0.90703 | 0.62234 |
| random_clifford | 64 | 14.521 | 2.5684 | 2.0555 | 14.529 | 2.8377 | 2.315 |
| random_clifford | 128 | 121.39 | 15.059 | 15.697 | 121.4 | 15.853 | 16.441 |
| random_clifford | 256 | — | — | — | — | — | — |
| random_clifford_t | 3 | 0.0080187 | 0.18578 | 0.15267 | 0.0086709 | 0.20339 | 0.17075 |
| random_clifford_t | 8 | 0.02686 | 0.21848 | 0.15193 | 0.027784 | 0.24642 | 0.18032 |
| random_clifford_t | 16 | 0.10583 | 0.33591 | 0.24311 | 0.10766 | 0.39632 | 0.30284 |
| random_clifford_t | 32 | 0.5277 | 0.59668 | 0.45401 | 0.53125 | 0.73023 | 0.58569 |
| random_clifford_t | 64 | 6.9228 | 1.5514 | 1.442 | 6.9299 | 1.9521 | 1.8305 |
| random_clifford_t | 128 | 86.393 | 6.2733 | 6.7857 | 86.407 | 7.8892 | 8.0184 |
| random_clifford_t | 256 | — | — | — | — | — | — |
| random_t_heavy | 3 | 0.0043438 | 0.17118 | 0.13476 | 0.004985 | 0.18648 | 0.14969 |
| random_t_heavy | 8 | 0.017312 | 0.19515 | 0.15479 | 0.018226 | 0.21963 | 0.17866 |
| random_t_heavy | 16 | 0.045142 | 0.25215 | 0.19464 | 0.046839 | 0.29487 | 0.23514 |
| random_t_heavy | 32 | 0.13917 | 0.5049 | 0.43298 | 0.14278 | 0.59631 | 0.52088 |
| random_t_heavy | 64 | 0.48037 | 0.72407 | 0.63724 | 0.49078 | 0.89519 | 0.79949 |
| random_t_heavy | 128 | 1.8576 | 1.2218 | 1.1695 | 1.8719 | 1.5787 | 1.5235 |
| random_t_heavy | 256 | — | — | — | — | — | — |
| spider_fusion | 2 | 8.97e-05 | 0.0083459 | 0.0081146 | 0.0001797 | 0.015421 | 0.01517 |
| spider_fusion | 16 | 0.0005546 | 0.0091375 | 0.0089903 | 0.000856 | 0.022235 | 0.021743 |
| spider_fusion | 128 | 0.011932 | 0.014216 | 0.013254 | 0.013991 | 0.065937 | 0.064743 |
| spider_fusion | 512 | 0.15531 | 0.030663 | 0.030376 | 0.19076 | 0.23719 | 0.2334 |
| spider_fusion | 1024 | 0.60737 | 0.049038 | 0.059686 | 0.62861 | 0.46442 | 0.47838 |
| spider_fusion | 4096 | 10.602 | 0.17018 | 0.17239 | 10.671 | 2.2742 | 2.0119 |
| spider_fusion | 16384 | — | — | — | — | — | — |
| supplementarity | 2 | 8.16e-05 | 0.010422 | 0.0028037 | 0.0001698 | 0.017374 | 0.0096803 |
| supplementarity | 16 | 0.000142 | 0.0099009 | 0.0029056 | 0.0003163 | 0.0186 | 0.011265 |
| supplementarity | 128 | 0.0009576 | 0.027226 | 0.019671 | 0.0018526 | 0.047918 | 0.040639 |
| supplementarity | 512 | 0.0064929 | 0.26415 | 0.21178 | 0.0097476 | 0.32978 | 0.27361 |
| supplementarity | 1024 | 0.018381 | 0.9939 | 0.82182 | 0.024624 | 1.1291 | 0.94122 |
| supplementarity | 4096 | 0.32013 | 15.713 | 14.53 | 0.34563 | 16.18 | 15.042 |
| supplementarity | 16384 | — | — | — | — | — | — |

Individual seed/configuration medians: [timings.csv](timings.csv). All trials, checks and timings: [observations.jsonl](observations.jsonl). Exact source archive: [sources.zip](sources.zip). Regressions and all 223 correctness fixtures: [validation report](../validation/REPORT.md).

## Errors

- {'family': 'bialgebra', 'size': 512, 'seed': 0} hybrid rep=0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.27GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- {'family': 'bialgebra', 'size': 512, 'seed': 0} cypher rep=0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 26.33GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- {'family': 'bialgebra', 'size': 1024, 'seed': 0} hybrid rep=0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 25.60GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
- {'family': 'bialgebra', 'size': 1024, 'seed': 0} cypher rep=0: TransientError: {code: Memgraph.TransientError.MemgraphError.MemgraphError} {message: Memory limit exceeded! Attempting to allocate a chunk of 4.00GiB which would put the current use to 25.71GiB, while the maximum allowed size for allocation is set to 23.47GiB.}
