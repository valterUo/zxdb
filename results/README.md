# Comprehensive ZXdb evaluation

The repository has one evaluation entry point: `python evaluate.py`. It runs regression tests, all eleven rule families, full-reduction correctness inputs, and paired PyZX benchmarks. Every run stores raw observations, source hashes, environment details, a CSV summary and a report together.

## Current work: native boundary pivots and zero factors

The native path now includes both Pauli and general boundary pivot, removing the
remaining boundary-pivot Cypher searches. Exhaustive neighbor-group, phase,
wire-type, overlap and invalid-candidate checks passed, as did the 113 full-reduction
fixtures. Explicit zero scalar inputs and isolated odd-pi zero spiders are now
preserved by the evaluation loader and cleanup; both formerly failing zero cases
pass through control, optimized and native reduction.

Exploratory rewrite-only timings (one trial, seed 0, no warmup):

| Circuit | Qubits | Database seconds | PyZX seconds | Speedup |
|---|---:|---:|---:|---:|
| T-heavy | 64 | 0.386 | 0.458 | 1.18x |
| Local layered | 64 | 0.855 | 0.804 | 0.94x |
| Clifford+T | 64 | 1.402 | 8.237 | 5.88x |
| Local layered | 128 | 5.214 | 7.855 | 1.51x |

A 64-qubit Clifford+T output was checked algebraically against the original
unitary circuit: composing with the adjoint reduced to 64 correctly ordered
identity wires and a provably nonzero symbolic scalar. A separate comparison to
the PyZX reduced output also yielded an identity certificate. These checks ran
outside the rewrite timer. The larger layered-circuit proof search was stopped after more than ten minutes and remains inconclusive.
This is progress toward the goal, not a completed repeated performance campaign.

[Measurements](native-boundary/profile.json), [certificate records](native-boundary/certificates.json),
[environment and source hashes](native-boundary/environment.json),
[native validation](native-boundary/validation.log), and
[zero-factor checks](native-boundary/zero-validation.log) are retained together.
The campaigns below describe earlier revisions and are not pooled with these trials.

## Latest optimization: grouped supplementarity

The native implementation now groups equal open/closed neighborhoods instead of running pairwise Cypher supplementarity joins. All native regression checks passed, including 113 full-reduction fixtures, supplementarity fixtures, batching, simple-edge rejection and graph isolation. The eight evaluation-contract tests also passed.

A separate exploratory paired profile used three repetitions at 64 qubits, seed 0, with alternating grouped/Cypher order and no warmup. Imports and exports were excluded. Large tensor checks remained inconclusive. This improves the implementation but does not yet establish the full performance goal.

| Circuit | Full reduction improvement over previous native path | Supplementarity improvement | New database speedup versus PyZX |
|---|---:|---:|---:|
| random_t_heavy | 1.57x | 48.5x | 0.59x |
| layered_local | 1.05x | 27.4x | 0.11x |
| random_clifford_t | 1.20x | 55.3x | 0.60x |

[Raw paired timings](native-supplementarity/paired.json), [source/environment hashes](native-supplementarity/environment.json), and [validation log](native-supplementarity/validation.log) are saved together. Pivot/boundary-pivot work remains the next measured bottleneck. The comprehensive campaigns below predate this latest change; their measurements remain evidence for their recorded revisions.

## Conclusions

- The native campaign finished: **60 regression tests passed**; **218/223 correctness inputs verified**, three were inconclusive and two failed because zero scalars are not preserved.
- **27/30 native performance cells completed**, with **10 rewrite-only timing wins**. Two connection losses and one timeout invalidated the other three cells. These are recorded, not retried into successes.
- Large native benchmark tensors were inconclusive. Small rule/full-reduction checks support the implementation, but do not constitute a tensor proof for each large timing input.
- Database speedups depend on circuit structure and scale. Dense Clifford cases benefit strongly; T-heavy and local layered cases favor PyZX at the measured sizes. This is a research demonstrator, not a universal performance or production-readiness claim.

## Native full reduction versus PyZX

The C++ procedures execute interior pivot, local complementation and gadget pivot inside Memgraph. Remaining rules use Cypher. Both engines start from the same graph; rewrite timing excludes import/export. Each cell has one warmup and three measured repetitions, with alternating engine order. Speedup is PyZX time / database time; above one favors the database.

| Circuit family | Qubits | Complete seeds | Median speedup range across seeds | Timing wins |
|---|---:|---:|---:|---:|
| random_clifford | 32 | 3/3 | 2.272ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“2.454x | 3 |
| random_clifford | 64 | 2/3 | 4.916ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“6.484x | 2 |
| random_clifford_t | 32 | 2/3 | 0.518ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.522x | 0 |
| random_clifford_t | 64 | 3/3 | 0.957ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“1.352x | 1 |
| random_t_heavy | 32 | 3/3 | 0.174ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.335x | 0 |
| random_t_heavy | 64 | 2/3 | 0.281ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.302x | 0 |
| layered_local | 32 | 3/3 | 0.199ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.245x | 0 |
| layered_local | 64 | 3/3 | 0.163ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“0.178x | 0 |
| layered_global | 32 | 3/3 | 0.219ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“1.765x | 2 |
| layered_global | 64 | 3/3 | 0.519ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“4.575x | 2 |

The 128-qubit exploratory Clifford+T trial took 70.75 seconds in the database and 145.70 seconds in PyZX (2.06x). This was a single run without a warmup or large-tensor verification, so it is separate from the repeated campaign. See [native-profile.json](native-profile.json).

## Coverage of every implemented rule

The earlier comprehensive Cypher campaign used three repetitions after a warmup over disjoint copies, growing neighborhoods and circuits. The following table shows its optimized variant. The control variant and every individual cell remain in the detailed report. These measurements predate the native campaign; the two runs are not pooled. Timing wins below can also have inconclusive tensor evidence.

| Rule | Complete performance cells | Rewrite-only timing wins |
|---|---:|---:|
| spider_fusion | 5/6 | 1 |
| identity | 6/6 | 1 |
| lcomp | 6/6 | 1 |
| pivot | 6/6 | 1 |
| pivot_gadget | 5/6 | 1 |
| pivot_boundary | 6/6 | 0 |
| gadget_fusion | 6/6 | 0 |
| bialgebra | 6/6 | 1 |
| hadamard_cancel | 6/6 | 0 |
| supplementarity | 5/6 | 0 |
| copy | 5/6 | 0 |
| full_reduce | 41/45 | 0 |

Across both Cypher variants, 209/222 performance cells completed. Its correctness results were 435 verified, six inconclusive, four failed zero-map checks and one execution error. The numerical verifier was reviewed conservatively; the report retains that provenance.

## Reproduce and inspect

```sh
python evaluate.py
python evaluate.py --variants native --full-reduce-only --qubits 32 64 --depth-factor 20 --seeds 3 --repetitions 3 --output results/native-new
```

Use an empty dedicated Memgraph instance. Compile the optional native module as described in [its README](../zxdb_qm/README.md). The second command still runs all correctness inputs; only its performance grid is restricted to full reduction. Failed checks produce a nonzero exit code after results are saved.

- [Complete Cypher report](full/REPORT.md) and [raw observations](full/observations.jsonl).
- [Complete native report](native/REPORT.md), [raw observations](native/observations.jsonl), [regression log](native/regressions.log), and [compiled-module provenance](native/module-build.json).
- [Historical campaign summaries](historical-summary.json) and [prior audit](../evaluation/audit-history.md).

The pre-cleanup files and obsolete raw results are preserved in the independent sibling repository `zxdb-backup-20261026`. The main repository retains the implementation, one evaluation runner, reusable fixtures and regression tests.
