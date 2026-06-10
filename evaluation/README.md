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

The 100-graph evaluation (`eval_full_reduce.py`) covers random CNOT+H+T
circuits (2-4 qubits, depths 6-14), Clifford-only circuits, T-heavy and
H-heavy circuits, phase-gadget graphs and 22 corner cases reused from the
per-rule suite. Status: **100/100 tensor-equivalent to the original graph,
92/100 isomorphic to pyzx's exact output** (isomorphism is not required —
different but individually sound match orders can yield different normal
forms), ~0.4 s/graph.

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

## How a case is judged

Each case builds a small pyzx graph (≤ 3 qubits so tensor contraction stays
fast), loads it into the DB, applies the DB rule and the pyzx rule, and
compares at three levels:

1. **Degree sequence** (`degseq`) — sorted vertex degrees of DB vs pyzx result.
2. **Graph isomorphism** (`iso`) — networkx VF2 with node type/phase (mod 2)
   and edge type matchers.
3. **Tensor equivalence** (`db_sem` / `pyzx_sem`) — the result's tensor against
   the ORIGINAL graph, up to global scalar, with a fallback over boundary
   permutations (the DB round-trip does not always preserve qubit order).

A case passes when the results are isomorphic, the DB holds no parallel
edges/self-loops (the exporter silently drops both, so the harness queries the
DB directly), and the DB result is semantically equal to the original. Cases
flagged `require_iso=False` (random circuits, overlapping match order) are
judged on tensor equivalence only. If pyzx itself produces a semantically
wrong result on a non-graph-like input (GIGO), matching pyzx exactly counts
as reference parity (e.g. `copy :: simple_edge_leaf`).

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
