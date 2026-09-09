# Interior pivot crash investigation (2026-09-08)

## Confirmed incident

The user's `results-study` campaign stopped on case `49324f2631b8374c`:
`random_t_heavy`, 16 qubits, depth 80, seed 4. The saved input has 127 vertices,
126 edges and SHA256
`39e31e7890e5ccdea208455fb558fab8c97c934979769261fb206b348a8fe392`.
The ten preceding blocks completed. No timing trial was saved for the failed block.

The client lost its connection while executing **Pivot rule - two interior Pauli
spiders**, through `full_reduce -> interior_clifford_simp -> pivot_rule`.
Docker recorded exit code 139 at 10:16:47 UTC; Memgraph 3.6.2 was using analytical
storage. This was a native segmentation fault, not the campaign time limit.
The server log has no native backtrace. The later cleanup connection failure was
secondary. No claim of an out-of-memory failure is supported by this evidence.

## What was investigated

Two disposable servers used the exact existing Docker image
`sha256:f48fd751312ed5d81c6a50c469b0e4b2b3ca0cf11fce7774393340554229a586`.
The user's server was not restarted, reconfigured or used for mutating reproductions.

The original query completed repeated runs on fresh servers, both with and without
per-query graph snapshots. These included 100 repetitions forcing an edge-index
scan, 30 normal benchmark repetitions, 30 repetitions with default server settings,
and replay of the original eleven-block prefix. The crash was **not reproduced**
in these runs. Consequently its precise native cause is still unconfirmed.

`EXPLAIN` showed that the original server could select
`ScanAllByEdgeTypePropertyValue`, whereas the isolated server initially selected
a node-index scan. Both had the Wire(t) index; optimizer choice, not mere index
presence, differed. Forcing edge scans did not reproduce the original fault.

A first mitigation candidate using nested list comprehensions inside `reduce`
caused a reproducible planner-time crash on a small pivot motif and was discarded.
Memgraph's upstream [3.7.0 release notes](https://github.com/memgraph/memgraph/releases/tag/v3.7.0)
and [list-comprehension planner patch #3338](https://github.com/memgraph/memgraph/pull/3338)
document a related class of planner failures. This is not proof that the original
campaign crash had the same cause. The retained change introduces no nested list
comprehensions or reduce expressions.

## Retained mitigation

Each of the three interior-pivot edge-toggle subqueries previously streamed
`OPTIONAL MATCH (u)-[existing]-(v)` directly into deletion, then tested the deleted
relationship variable to decide whether to create a wire. It now aggregates
`collect(existing)` for the endpoint pair before any mutation, deletes from that
materialized list, and creates a wire only if the list was empty.

This drains the relationship expansion before deleting from the adjacency list
it was traversing, and avoids consulting the deleted relationship variable.
Endpoint selection, neighbor partitions, phases, graph scope and rule order are
unchanged. It remains a Cypher rewrite. This is targeted defensive hardening;
successful tests do not establish that the original intermittent native crash
has been eliminated.

The research runner now records the last attempted Cypher query on failure and
records cleanup errors separately without masking the primary exception.

## Validation and follow-up

`tests/test_pivot_materialization.py` exercises all 16 combinations of empty/nonempty
exclusive/shared neighbor groups and existing/absent neighbor edges. Each checks
the original tensor, one pivot application, and fixpoint stability. It also runs
the reported 16-qubit circuit six times as a liveness regression (not a dense
32-open-leg semantic proof). The research suite tests and existing rule/full-reduce
evaluations are run separately; logs are under `evaluation/audit_results/`.

Validation completed: 17 research-suite tests, both new pivot test methods
(16 semantic subcases and six large-input repetitions), 108/108 legacy rule cases,
and 113/113 legacy full-reduce cases passed. The legacy evaluations have the
oracle limitations documented in the research guide; they are additional
regressions, not a proof of native crash safety.

Run live regressions on a dedicated empty server:

```powershell
$env:ZXDB_RESEARCH_LIVE='1'
python -m unittest tests.test_pivot_materialization tests.test_research -v
```

For further crash isolation on a disposable server, `evaluation.debug_pivot_crash`
defaults to port 7688 and the saved failing input. It can capture the graph/query
before each counted rewrite, or omit extra reads using `--no-snapshots`. Do not
use its `--edge-scan` option on a shared server: that diagnostic deliberately
removes node indexes to force a different execution plan.

Do not merge timings from before and after this change as one implementation.
Use a new campaign output directory. If the crash recurs, retain its raw block
record and server core/backtrace; no automatic restart or skipped failure should
be treated as a passing experiment.
