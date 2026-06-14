# Rule evaluation suite

Critical correctness evaluation of every rewrite rule in `zxdb/main_queries.json`
against its pyzx counterpart, using small adversarial corner cases plus random
average cases — and of the composed `full_reduce` pipeline.

## Running

```
python -m evaluation.run_eval                  # all rules (~104 cases)
python -m evaluation.run_eval pivot lcomp      # selected rules
python -m evaluation.eval_full_reduce          # full_reduce on 100 graphs
python -m evaluation.eval_full_reduce --quick  # ~20 graphs
python -m unittest tests.test_full_reduce      # fast unittest subset
```

Requires the Memgraph instance on `bolt://localhost:7687`.

**Run with the repo's `myenv` environment (pyzx 0.10.3).** The reference
semantics are version-sensitive:

- pyzx 0.10 `*_simp` rewrites are `Rewrite` objects with INCREMENTAL
  matching: a single call can stop before the global fixpoint (observed:
  `pivot_boundary_simp` on a circuit with several disjoint pattern copies
  stops after one). The DB rules iterate to a fixpoint, so every pyzx
  reference here is wrapped in `pyzx_fixpoint` (in `utils.py`), which calls
  the rule until the graph stops changing. The legacy `tests/test_*` files
  use the same wrapper.
- pyzx 0.10 changed the SEMANTICS of two rules relative to 0.9, and the DB
  queries follow 0.10:
  - **copy_simp**: the arity-1 Pauli leaf is copied through its neighbor as
    one new toggled-color spider per remaining wire (same wire type); no
    phase merging into neighbors, no isolated-vertex cleanup. Across a
    Hadamard wire the colors must match; across a simple wire they must
    differ.
  - **bialg_simp**: fires on PAULI (not just phase-0) Z-X centers joined by
    one simple wire whose other neighbors are phase-0 spiders of the swapped
    color; every other wire of each center receives a NEW spider of the
    opposite color carrying the opposite center's phase, and the two new
    groups are connected completely bipartitely ("Bialgebra rule" +
    "Bialgebra connect" queries).
- pyzx 0.10 `full_reduce` also calls `copy_simp` and `supplementarity_simp`
  in its main loop and ends with `remove_isolated_vertices()`;
  `ZXdb.full_reduce` mirrors that.

**Zero diagrams:** randomly generated phase-gadget graphs can be the zero
map (all-zero tensor). Rewrites are then only correct "up to a zero scalar",
which the DB does not track, so the tensor check is indeterminate; the
harness detects this and falls back to the structural comparison. (The check
also guards against the `compare_tensors` false positive where an all-zero
tensor "equals" anything.)

**Multigraph backend (pyzx >= 0.10):** rewrites such as lcomp can leave
PARALLEL Hadamard wires in place; the DB rules cancel them eagerly (Hopf).
Both forms are semantically equal. Where a legacy test compares structure
directly, the pyzx reference is wrapped in `pyzx_fixpoint_normalized`
(`utils.py`), which additionally runs pyzx's own `hopf_simp` and
`remove_self_loop_simp`.

**Legacy test sizes:** `test_pivot_gadget_rule` and `test_lcomp_rule` used
to expand their circuits 2^14 / 2^17-fold, which made the comparison
machinery run for half an hour or get OOM-killed; both now use the same
2^2 expansion as the other tests.

**0.9 idiom fixed:** `g.add_edge(g.edge(u, v), t)` raises `KeyError` on the
0.10 multigraph backend (`edge()` looks up an existing edge); call sites in
`zxdb/generate.py` and `zxdb/pyzx_utils.py` now pass the `(u, v)` tuple.

## full_reduce

`ZXdb.full_reduce(graph_id)` mirrors pyzx exactly
(`pyzx/simplify.py::full_reduce`): `interior_clifford_simp` (spider fusion,
`to_gh` color change, then rounds of identity removal / spider fusion / pivot
/ local complementation), one `pivot_gadget_simp`, then rounds of
`clifford_simp` + `gadget_simp` + `interior_clifford_simp` +
`pivot_gadget_simp` until the gadget rules stop firing. `to_gh` is implemented
by the new "Color change edge toggle" / "Color change spiders" queries (each
wire toggles once per X endpoint, then X-spiders recolor to Z). A `max_rounds`
cap turns potential non-termination into a visible error.

The `eval_full_reduce.py` evaluation (108 graphs) covers random CNOT+H+T
circuits (2-4 qubits, depths 6-14), Clifford-only circuits, T-heavy and
H-heavy circuits, phase-gadget graphs, **structured circuits** (QFT on 2-4
qubits, structured Clifford+T, and algebraic circuit identities that should
reduce to the identity), and 22 corner cases reused from the per-rule suite.
Status: **108/108 pass** (107 tensor-equivalent to the original graph; the one
remaining is a randomly-generated zero-map phase-gadget diagram where the
tensor check is indeterminate — see "Zero diagrams" above — and falls back to
the structural comparison). ~0.6 s/graph. Isomorphism to pyzx's exact output
holds for 91/108 and is not required (different but individually sound match
orders yield different normal forms).

Two pipeline-level bugs surfaced only under composition and were fixed:

- **Spider fusion null crash** — when a fused endpoint had no neighbors
  besides its partner, the unguarded `CREATE (merged)-[...]->(x)` received a
  null `x` ("Expected a vertex for 'x', but got null"). Both reconnection
  steps are now FOREACH-guarded.
- **Mixed parallel pair lost during fusion** — a fusable pair connected by
  BOTH a simple and a Hadamard edge must fuse to a spider with an extra pi
  (the Hadamard edge becomes a self-loop); the fusion query silently dropped
  the second edge. Parallel-edge normalization now runs inside the spider
  fusion loop (shared `_normalize_parallel_edges` helper), not only in
  `remove_identities`.

## How a case is judged — tiered verification

Each case loads a graph into the DB, applies the DB rule and the pyzx rule,
and judges correctness at the **strongest feasible level**, recorded as
`level` (printed per case and tallied in the summary). In order of authority
(`evaluation/harness.py::_verdict`):

1. **tensor (db vs pyzx)** — both results are reduced and small, so this is
   the common, decisive check. Equal up to a global scalar ⇒ pass; different
   ⇒ a real failure. Tensor equality is invariant to non-confluence, so DB and
   pyzx producing structurally different normal forms is fine here.
2. **tensor (db vs original)** — only if (1) is infeasible, compares to the
   input circuit.
3. **sampled tensor** — if a full dense contraction is infeasible (high
   treewidth), fix every open leg to random 0/1 values, closing the network so
   it contracts to a scalar amplitude at far lower cost, and compare amplitude
   vectors over many random samples up to one global scalar.
4. **isomorphism** — db vs pyzx structure (networkx VF2, phases mod 2).
5. **degree sequence** — weakest fallback.
6. **unverified** — no method could verify (tensor infeasible *and* structures
   differ under non-confluence). Reported honestly; not counted as a failure
   since there is no evidence of a bug.

A case also requires the DB to hold no parallel edges / self-loops (the
exporter silently drops both, so the harness queries the DB directly). If pyzx
itself is wrong on a non-graph-like input (GIGO), the DB matching pyzx exactly
at tensor level still passes (e.g. `copy :: simple_edge_leaf`).

### The tensor engine (quimb) and why there is no permutation search

Tensor comparison uses **quimb** (`_quimb_tensor` / `_quimb_equal`), not
pyzx's `to_tensor`. pyzx contracts left-to-right and builds enormous
intermediates on gadgetized / high-degree graphs (a degree-35 spider alone
would need a 512 GiB dense array) and simply hangs. The quimb engine:

- **decomposes high-degree spiders** into chains of degree-3 spiders, so no
  single tensor exceeds 2³;
- uses an **optimized contraction path** (opt_einsum greedy) and refuses
  graphs whose largest intermediate would exceed 2²⁸ (`WIDTH_CAP`), bailing in
  milliseconds instead of hanging;
- handles Z/X spiders, Hadamard edges and H-boxes.

Boundary open legs are named by **input/output position**, and the DB
round-trip now preserves I/O ordering (`ZXdb` stores an `io_index` on each
boundary at import and sorts by it on export), so db and pyzx tensors line up
directly — the old `(n!)²` boundary-permutation search is gone.

Net effect: every per-rule case (107) and every full_reduce case (112,
including **wide circuits up to 12 qubits**) is verified at the tensor /
sampled-tensor level — the old "~4 qubit / no permutation" limits are removed.
Deep wide random circuits (q7-q14 in `benchmark_perf --verify`) remain
"unverified": their full_reduce'd form has high treewidth so exact contraction
is infeasible for *any* method (pyzx included) — a verification-cost limit,
not a correctness gap. The same rules are tensor-verified on wide *shallow*
circuits up to 12 qubits, so qubit count itself is not a concern.

## Rule ↔ pyzx mapping

| evaluation name   | ZXdb method                  | pyzx                     |
|-------------------|------------------------------|--------------------------|
| spider_fusion     | `spider_fusion`              | `zx.spider_simp`         |
| identity          | `remove_identities`          | `zx.id_simp`             |
| hadamard_cancel   | `hadamard_cancel`            | `zx.id_simp` (subset)    |
| lcomp             | `local_complementation_rule` | `zx.lcomp_simp`          |
| pivot             | `pivot_rule`                 | `zx.pivot_simp`          |
| pivot_gadget      | `pivot_gadget_rule`          | `zx.pivot_gadget_simp`   |
| pivot_boundary    | `pivot_boundary_rule`        | `zx.pivot_boundary_simp` |
| gadget_fusion     | `phase_gadget_fusion_rule`   | `zx.gadget_simp`         |
| bialgebra         | `bialgebra_simp`             | `zx.bialg_simp`          |
| supplementarity   | `supplementarity_simp`       | `zx.supplementarity_simp`|
| copy              | `copy_simp`                  | `pyzx.simplify.copy_simp`|

## Bugs found and fixed (2026-06-10)

Status before fixes: many corner cases failed or hung. After fixes:
**104/104 cases pass.**

**Remove identities with refactor** — used Memgraph `refactor.collapse_node`,
which copies the NODE's properties onto the new relationship: the new edge's
`t` came from the spider type, not the ZX edge algebra. X-identities produced
Hadamard wires, s+h combos produced simple wires (h+h only worked by
accident). Rewritten as an explicit query implementing s·s=s, s·h=h, h·h=s,
including the both-edges-to-same-neighbor self-loop case. Three new
normalization queries mirror pyzx `add_edge_table` for parallel edges between
same-colored spiders: H-pairs cancel, s∥h → s with +π on one endpoint,
s∥s → single s.

**Supplementarity type 1 / type 2** — type 1 missed the `(α−β) % 2 == 1`
match condition; type 2 matched ANY adjacent non-Clifford pair with equal
external neighborhoods (unsound — tensor check failed) instead of requiring
sum even or difference odd. Phase propagation is now gated on the sum
condition exactly as in pyzx `apply_supplementarity`.

**Local complement** — matched `phase = -0.5` which never occurs (DB stores
3/2), so −π/2 spiders never fired; did not check the spider is interior, so a
boundary-attached center fired and `DETACH DELETE` silently severed the
boundary wire (unsound); a degree-1 center killed the query row mid-pipeline
so the center was never deleted; and the driver loop hung forever when no
match existed. Fully rewritten (one application per call, driver loops).

**Pivot (two interior / single)** — phase propagation was SWAPPED (a's
exclusive neighbors received a's phase; pyzx copies b's phase onto them —
only visible with mixed 0/π phases); interiorness was not checked
(boundary-adjacent spiders fired unsoundly); no LIMIT 1. The "single interior
Pauli spider" query only handled the degree-2 special case of pyzx's boundary
pivot; it now implements the general boundary pivot (boundary spider with
arbitrary extra neighbors).

**Pivot gadget** — no non-Pauli requirement on z_alpha (fired on plain pivot
pairs), no degree-1 guard (re-gadgetized phase-gadget tips forever), clique
edges CREATEd instead of toggled (parallel edges when neighbors already
connected), no LIMIT 1, and a nonstandard axis/tip phase convention
(axis=π, tip=±α) that is semantically equal but not isomorphic to pyzx's
(axis=j, tip=α). Rewritten.

**Pivot boundary** — same toggle and phase-convention fixes; added missing
neighbor-type guards; one application per call.

**Gadget fusion** — fused gadgets with ANY leaf phase (pyzx requires
non-Clifford leaves); ignored axel phases entirely (an axel carrying π must
negate that gadget's contribution); never zeroed the surviving axel; total
phase not reduced mod 2; the grouping key was collected without ORDER BY
(nondeterministic grouping). Rewritten + new "Gadget axel normalization"
query mirroring pyzx's inline normalization of single gadgets with axel π.

**Bialgebra** — the old labeling/simplification queries implement a biclique
→ edge CONTRACTION, the OPPOSITE direction of pyzx `bialg_simp` (edge →
biclique on phase-0 Z–X pairs whose neighbors are all phase-0 spiders of the
swapped color). New pyzx-faithful "Bialgebra rule" query added (with Hopf
toggle for already-connected neighbor pairs); the old queries remain in the
JSON but are no longer used by `bialgebra_simp`.

**Spider fusion / copy / phase updates everywhere** — phases are now
normalized mod 2 when written (9/4 stored instead of 1/4 broke later
exact-phase matches like `phase = 0.5`).

**Driver methods** (`zxdb.py`) — `remove_identities`,
`local_complementation_rule`, `pivot_rule`, `pivot_gadget_rule`,
`pivot_boundary_rule`, `phase_gadget_fusion_rule`, `bialgebra_simp` now loop
their single-application queries to a fixpoint and return application counts;
several previously ran one pass only, and lcomp/bialgebra could spin forever
on `result.single()` returning `None`.

## Known accepted divergences

- `hadamard_cancel :: chain3` — the rule only collapses even all-H paths and
  leaves one identity behind on odd chains; `remove_identities` finishes the
  job. Semantically equal (judged by tensor).
- `copy :: simple_edge_leaf` — pyzx `copy_simp` itself mis-rewrites a copy
  leaf attached by a SIMPLE edge (it assumes graph-like input). The DB
  matches pyzx exactly; both are "wrong" relative to the original tensor.
- The exporter (`export_graphdb_to_zx_graph`) silently drops parallel edges
  and self-loops, and its spring-layout positions break pyzx `to_tensor`
  (the harness rebuilds position-free copies with BFS rows).

## Performance vs pyzx (`evaluation/benchmark_perf.py`)

```
python -m evaluation.benchmark_perf            # default sizes (fast, <2 min)
python -m evaluation.benchmark_perf --big      # adds larger sizes
python -m evaluation.benchmark_perf --profile  # per-rule round-trip/time
```

Wall-clock `full_reduce`, DB vs pyzx, on random CNOT+H+T circuits (one machine,
clean DB, warm):

| qubits/depth | input verts | DB full_reduce | round-trips | pyzx | ratio |
|--------------|-------------|----------------|-------------|------|-------|
| q4  d40  |   69 | 0.24s |  86 | 0.005s | 47x  |
| q6  d100 |  161 | 0.32s |  99 | 0.016s | 20x  |
| q8  d200 |  326 | 0.49s | 162 | 0.049s | 10x  |
| q10 d300 |  489 | 0.94s | 243 | 0.079s | 12x  |
| q12 d450 |  730 | 1.23s | 243 | 0.196s | 6.3x |
| q14 d650 | 1024 | 2.14s | 312 | 0.481s | 4.5x |

### Why pyzx is a hard baseline, and the one number that matters

`full_reduce` issues ~100-300 Cypher queries (one per fixpoint step), and on
this setup each costs ~0.8 ms of transport plus query execution. pyzx does the
whole reduction in-process with no serialization, in microseconds per rewrite.
So the DB carries a **fixed round-trip floor** (~0.2-0.3 s) that pyzx does not.

The decisive observation: the **round-trip count grows far slower than graph
size** (86 at 69 verts → 312 at 1024 verts, ~4x for ~15x the vertices), because
it tracks the *number of fixpoint iterations*, not the vertices. pyzx's per-run
cost grows super-linearly. So the DB/pyzx ratio collapses as graphs grow —
47x → 4.5x over the table above — and extrapolates to a crossover beyond
~2000-3000 vertices (where pyzx itself takes seconds). The DB approach is
structurally suited to *large* diagrams, not small ones.

### Optimizations applied (all correctness-preserving — 107/107 rule cases,
### 100/100 full_reduce graphs still pass)

- **Label indexes** on `:Node(t)` and `:Node(graph_id)` (created at startup)
  so every `MATCH (n:Node {t: ...})` is an index seek, not a full scan.
- **Single transaction per rule fixpoint.** Each rule's loop now runs all its
  iterations inside one `execute_write` (via `_run_count` / shared query
  text), instead of a new transaction per iteration.
- **Parallel-edge guard.** `_normalize_parallel_edges_tx` runs one cheap
  count query and skips the three normalization passes entirely when there are
  no parallel edges — the common case in the pipeline. Turns a fixed
  6-round-trip cost into 1.
- **Spider fusion restructure.** Normalize once up front, then only after a
  merge actually creates parallel edges (not before every fuse), and skip the
  Hopf pass on the terminal no-op iteration.
- **Batched identity removal.** One query removes every identity spider in a
  pass (re-checking preconditions per row for chains), instead of one per call.
- **Supplementarity guard.** Skip the quadratic pair-matching queries unless
  at least two non-Clifford spiders exist.

A *whole-pipeline* single transaction (collapsing full_reduce's ~30
transactions into one) was prototyped and is ~15% faster, but under
`IN_MEMORY_ANALYTICAL` it lets every create/delete delta accumulate until one
commit and destabilizes the server, so it is **disabled** in `full_reduce`
(the `_active_tx` machinery remains for callers that opt in). Correctness and
stability outrank the 15%.

### The hidden killer: leaked nodes

While benchmarking we found **917,157 stray nodes** in the database. Memgraph's
`IN_MEMORY_ANALYTICAL` mode has no transaction rollback, so any server crash or
container stop *mid-rewrite* leaves that rewrite's partially-created
intermediate nodes behind. Across many interrupted runs they accumulate; every
`MATCH (:Node)` then scans all of them, and the bloated snapshot makes each
restart heavier — a feedback loop that presents exactly as "the queries got
slow / the server keeps dying". Wiping the database (`ZXdb.wipe_database()`,
which batch-deletes and calls `FREE MEMORY`) restored both speed and
stability, and the full 100-graph suite then ran start-to-finish with no crash.

**Operational guidance:** keep one working graph in the DB. Call
`zxdb.node_count()` to check for bloat and `zxdb.wipe_database()` to reset
between unrelated workloads. `zx_graph_to_db(..., initialize_empty=True)`
already wipes on every load when it completes uninterrupted.

### To actually beat pyzx

The round-trip floor is the wall for small/medium graphs. The only way under it
is to remove the client-server boundary: run the fixpoint **inside** Memgraph
as a query module (MAGE / `mgp`) so the ~100-300 round-trips become one call.
That keeps the (verified) Cypher rewrite logic but executes it server-side.
This is the recommended next step and is left as future work because a faithful
port carries real correctness risk that must be re-validated against the suites
here.
