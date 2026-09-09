# Interpretation of the adaptive campaign — 2026-09-09

This analysis reads the saved plan, summary, campaign, recovery and raw block records. It does not rerun Memgraph, modify the benchmark or alter observations. All campaign sessions used one source version; current evaluation/database source hashes match that version. Two small PyZX-only profiling probes were run separately and are diagnostic, not campaign measurements.

## Coverage and empty figures

The plan contains 871 cases grouped into 451 cells (family, size, depth, with seeds pooled). Campaign outcomes: 560 complete, 67 both-engine timeout stops, 166 adaptive skips, 74 skips after predecessor case-budget exhaustion, and 4 case-budget exhaustions. Worker completion does not imply an acceptable paired performance cell: resource-limited trials and earlier failed attempts can remain in a completed worker.

260 cells meet the report's complete-pair criteria; 191 do not. Of 87 family/depth plot groups, 31 have no complete cell. Fifteen of these have no measured trials; sixteen contain measurements excluded from paired aggregation. The report generates figures for the entire planned grid, including unrun sizes. It sets paired timings to missing if any planned repetition/seed is missing, a trial or warmup has a resource limit, an attempt failed, or versions differ. Large-case semantic inconclusiveness alone does NOT suppress timing curves.

Example: layered_global, depth 128, width 2 has six successful measured pairs, but an earlier failed attempt suppresses its main timing/speedup point. Other widths have no accepted points, so the entire depth-128 main plot is empty. Its capacity fragment retains available measured timings, subject to its survivor-only interpretation.

The report also still plots circuit width at fixed depth, although the adaptive scheduler grows depth at fixed width. This spreads observations over many sparse pages. Capacity fragments exist but are not included in figures.tex. Presentation should instead show depth ladders at fixed width, explicitly show completion/timeout/crash status, omit all-unrun panels from the main document, and keep the full grid/status appendix. Do not replace censored times with the timeout threshold or compute finite speedups from them.

## Meaning of copies

A copies_RULE input is the disjoint union (tensor product) of N copies of the corresponding size-2 rule motif. Components have no connecting edges. N scales the number of opportunities while keeping each component's topology and degree bounded. These are synthetic rule-throughput workloads, not realistic full-reduction circuits. The benchmark invokes the named simplifier to convergence, not full_reduce. Full-reduction families are the random_* and layered_* families and have no planted motifs.

Some base generators already consist of disjoint patterns. For identity and spider_fusion, a copies motif itself contains two patterns, so comparisons at the same x-axis value do not imply equal input size. Compare input vertex counts/opportunities too; do not treat those families as independent evidence of different graph structures.

## Strong reduction-time results

Representative largest complete paired cells, using recorded median reduction times and the saved paired speedup estimator:

| Family | Copies | Vertices | ZXdb seconds | PyZX seconds | Paired speedup |
|---|---:|---:|---:|---:|---:|
| copies_identity | 2048 | 12288 | 0.0555 | 7.3985 | 133.2x |
| copies_lcomp | 4096 | 20480 | 0.1664 | 11.0048 | 66.1x |
| copies_spider_fusion | 2048 | 16384 | 0.1916 | 11.0656 | 57.7x |
| copies_pivot | 2048 | 20480 | 0.2622 | 7.1279 | 27.2x |
| copies_gadget_fusion | 8192 | 65536 | 0.4360 | 3.9978 | 9.2x |

These are sustained scale trends, not just one isolated crossing. Small instances generally favor PyZX because database fixed costs dominate. In the five cells above, ZXdb makes respectively 4, 3, 7, 4 and 5 Cypher submissions during reduction. Further unpaired database completions exist: copies_identity reaches 16384 copies with median reduction 0.544 seconds, copies_pivot 8192 with 1.117 seconds, and copies_spider_fusion 8192 with 0.762 seconds while PyZX has no completed measured attempt there. These are capacity observations under the protocol; they do not give exact speedups.

A separate cProfile probe on 1024 copies found:

- identity: 2048 remove_isolated_vertices calls, 6.717 seconds cumulative out of 6.976 seconds profiled total (96.3%), about 10.5 million vertex-degree calls;
- lcomp: 1024 remove_isolated_vertices calls, 2.595 seconds cumulative out of 2.830 seconds total (91.7%), about 4.7 million vertex-degree calls.

The installed PyZX 0.10.3 implementation scans all vertices in remove_isolated_vertices. These probes identify repeated global cleanup as a dominant cost on those inputs. Profiling changes absolute timing; these numbers must not replace campaign timings. The conclusion concerns this baseline implementation. A PyZX implementation with localized/deferred cleanup may reduce the observed gap.

## Negative and mixed patterns

Not all copies families benefit. At 256 copies, copies_pivot_boundary takes 4.109 seconds versus PyZX 0.0263 seconds and makes 258 database submissions. copies_bialgebra at 256 takes 3.925 versus 0.0656 seconds and makes 514 submissions. copies_pivot_gadget at 512 takes 4.454 versus 0.1903 seconds and makes 514 submissions. Repeated calls and scans remain an important limitation even when opportunities are independent.

Of 90 complete realistic full-reduction cells, 89 favor PyZX by the saved paired estimator. The sole ZXdb point-estimate win is random_clifford, 32 qubits, depth 128: 1.155x, with bootstrap interval 0.983–1.424, spanning parity. There is no convincing general full-reduction advantage in these data. Growing fan degree also does not reliably favor the database: pivot at fan 512 takes 1.040 versus 0.904 seconds; lcomp at fan 256 takes 1.302 versus 0.101 seconds. Several intermediate PyZX timings are irregular, so isolated crossover points need repetition/profiling before interpretation.

## Reduction time is not total processing time

Loading/reading is excluded from the plotted reduction times. For copies_gadget_fusion at 8192, ZXdb setup + reduction + read medians sum to 11.558 seconds versus PyZX copy + reduction of 4.250 seconds. The 9.2x reduction-only gain reverses on those recorded processing stages. For copies_identity at 2048, the corresponding sums are 1.724 versus 7.505 seconds, so a smaller advantage remains. Sums of stage medians are illustrative, not paired total-time estimators, and omit process startup/IPC, generation and validation. Publish separate resident-graph reduction and end-to-end results.

The 15-second database attempt budget includes loading and reading, while the PyZX process budget includes its own startup/import/copy/serialization protocol. Database queries also have a 10-second cap. Thus a timeout is not proof that reduction alone takes more than 15 seconds. RSS limits are not symmetric across the full systems. This study does not establish out-of-core or beyond-RAM scaling.

## Correctness, reliability and uncertainty

There are 1126 measured trials: 930 ok and 196 resource-limited. No measured tensor failure was recorded. Direct DB–PyZX tensor comparison verified 229 measured pairs; 897 were inconclusive, including unavailable outputs. Graph comparison found 469 equal, 321 different and 336 inconclusive; degree comparison found 619 equal, 311 different and 196 inconclusive. Graph/degree differences can arise from valid different reductions and are not by themselves semantic failures. They also warrant examining reduction quality separately from execution time.

Only 24 complete cells have all original-to-output tensor checks verified; 236 complete cells are semantically inconclusive under the bounded checks. Large speedup examples cannot be described as tensor-verified results. Rule-motif validation supplies local evidence, not a general proof for every measured graph.

The campaign logged 17 successful automatic recoveries and four worker wall-clock timeouts. ServiceUnavailable/defunct connections are present. This demonstrates recovery robustness but continuing server availability problems; these records alone do not establish the native crash cause. Stability belongs in the reported outcomes.

Rule inputs have one deterministic seed and two measured repetitions; narrow bootstrap intervals mostly describe repeat timing noise. Circuit cells have three seeds, sometimes identical inputs. Do not present these intervals as uncertainty over broad circuit populations or general machines. Increasing-size scheduling can also correlate size with machine drift; repeated independent campaigns and declared host/container resource settings would strengthen the study.

## Defensible paper claim

On controlled workloads containing many independent rewrite opportunities, batched Cypher execution substantially reduces rule-execution time relative to PyZX 0.10.3. The gains are rule-dependent and consistent with avoiding repeated global scans and database round trips. They do not extend to a general advantage in realistic full reduction in this campaign. Data movement, serial rewrite paths, timeouts and server stability remain material limitations.
