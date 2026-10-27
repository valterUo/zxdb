# ZXdb

A research prototype comparing ZX-calculus rewrites in Memgraph with PyZX.
The aim is a readable demonstration of when database batching helps, including
cases where database overhead makes it slower. This is not a production service.

All evaluation evidence is indexed in [results/README.md](results/README.md).
The pre-cleanup repository is preserved in the independent sibling directory
`zxdb-backup-20261026`.

## Run the complete evaluation

Use Python 3.12 and a **dedicated empty Memgraph instance** on
`bolt://localhost:7687`. No concurrent database writers may run during evaluation.
The runner uses analytical storage mode, as required by the rewrite queries.

```sh
python -m pip install -r requirements.txt
python evaluate.py
```

On this Windows checkout, use `myenv/Scripts/python.exe evaluate.py`.
Set `ZXDB_URI`, `ZXDB_USER`, and `ZXDB_PASSWORD` for a different connection.

One command runs implementation regressions, 223 correctness inputs against both
control and optimized database implementations, then a fixed scalability grid
against PyZX 0.10.3. It writes everything under `results/latest/`:

- `REPORT.md`: correctness, timings, failures and interpretation.
- `observations.jsonl`: every trial, including warmups and failed trials.
- `summary.json` and `summary.csv`: paired median speedups and observed ranges.
- `environment.json`: versions, machine details, configuration and source hashes.
- `regressions.log`: detailed regression results.

`python evaluate.py --quick --output results/quick-new` keeps **all correctness
inputs** but uses a smaller performance grid and one repetition. The full grid
uses three repetitions after one warmup. Choose a fresh output directory for a
new run; results are never silently overwritten. A failed check gives a nonzero
exit code after the report is saved. Inconclusive checks are reported separately.

For a dedicated local Docker container, `python evaluate.py --restart-container
memgraph --output results/full` can continue after a server crash. The failed
trial remains an error and its performance cell receives no speedup. This option
also enables a database watchdog (`--trial-timeout 120`, in seconds). After an
execution error, remaining repetitions of that cell are left missing. Without
this option, database loss stops the run and saves a partial report. Memgraph
3.6.2 crashed during exploratory full-reduction runs on this machine.

After an interrupted run, repeat the same command and settings with `--resume`.
Completed observations and failures are retained; failed cells are not retried.
Changing workload, oracle or rewrite sources prevents continuation. Runner
versions are snapshotted per attempt, and timing cells spanning attempts receive
no aggregated speedup. Regression execution is capped at five minutes.

For circuit benchmarks, `--certificate-timeout 120` optionally attempts a bounded
algebraic identity proof when tensor verification is inconclusive. This relies on
the input being unitary by circuit construction and is never applied to arbitrary
rule fixtures. Proofs run outside rewrite timing and the database watchdog, and
are cached by input/output hashes. A timeout remains inconclusive.

## What is measured

All eleven implemented rule families are included: spider fusion, identity and
Hadamard cancellation, local complementation, ordinary/gadget/boundary pivot,
phase-gadget fusion, bialgebra, copy, and supplementarity. `full_reduce` is tested
on adversarial compositions, random Clifford and Clifford+T circuits, T-heavy
circuits, QFT and layered circuits. Graph isolation, parallel-edge normalization,
and batching regressions are retained in `tests/`.

The performance grid separates disjoint pattern copies, growing neighborhoods,
and actual circuits. Each timed trial starts from the same input graph; engine
order alternates. PyZX runs its native rule API, repeating only its three
single-batch APIs until completion. Its `full_reduce` is called once.

Speedup is **PyZX seconds / database seconds**: values above one favor ZXdb.
Rewrite-only and import/rewrite/export timings are reported separately. Import
and export use the small graph adapter in `evaluation/core.py`, not the legacy
JSON APIs or visualization layout. There is no claim that a synthetic rule win
implies faster full circuit reduction. Large tensors can exceed the oracle's
resource bounds; those measurements remain explicitly unverified. Near-zero or
nonfinite numerical tensors are inconclusive unless an exact zero factor gives
a decisive comparison. The saved report documents the conservative review of
evidence collected before this numerical-checker correction.

## Reading the code

1. `evaluate.py`: the single evaluation entry point and report generation.
2. `evaluation/cases.py` and `circuits.py`: correctness fixtures.
3. `evaluation/patterns.py` and `workloads.py`: scalable deterministic inputs.
4. `evaluation/core.py` and `oracle.py`: graph I/O, reference calls and verification.
5. `zxdb/zxdb.py`: Python rewrite loops; `zxdb/main_queries.json`: Cypher rules.
6. `zxdb_qm/zxqmcpp.cpp`: scoped native rewrite loops.

`zxdb_qm/` provides optional native database procedures for interior pivot, local
complementation, boundary pivot, gadget pivot and grouped supplementarity. See its [build instructions](zxdb_qm/README.md).
Use the same evaluation entry point for larger circuit comparisons:

```sh
python evaluate.py --variants native --full-reduce-only --qubits 32 64 128 --depth-factor 20 --seeds 3 --output results/native
```

This still checks all correctness inputs before benchmarking. `--variants`
also accepts `control optimized native` for a direct three-way comparison.
Native procedures require compilation and a database restart before evaluation.
The default evaluation needs no compiled module.

Known limitations include incomplete scalar tracking, standalone Cypher
supplementarity preconditions, and the legacy JSON import/export APIs. The
evaluation adapter preserves explicit zero scalar factors as isolated pi spiders,
and cleanup retains those zero factors. General scalar tracking is still absent. See `evaluation/audit-history.md` for the prior audit, and the
new report for evidence on this revision. Historical campaign summaries are kept
in `results/historical-summary.json`; raw obsolete runs live in the backup repo.
