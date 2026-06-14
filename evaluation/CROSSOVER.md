# Where does the database `full_reduce` beat pyzx? A circuit-family study

> **Refined by [DIMENSIONS.md](DIMENSIONS.md) (controlled OFAT study).** This
> family study found QAOA crosses over and attributed it to "wide, shallow,
> sparse." The follow-up dimensional study shows the real drivers are an
> **inverted-U in connectivity** (locality `span` and density `degree`) plus
> gadget content — *not* width/depth, and *not* treewidth. Read DIMENSIONS.md for
> the current conclusion; this file remains as the family-level evidence.

`python -m evaluation.crossover_search [family ...]`

The per-rule study ([SCALABILITY.md](SCALABILITY.md)) established the rule: the
DB wins when a rewrite is **one large operation** (high-degree spider, O(N) or
O(N²) work in a single Cypher query) and loses when reduction is **many small
operations** (each ≈ one network round-trip). Random deep circuits are the worst
case — they fragment into hundreds of tiny pivots, so the DB carries a ~1.5–2×
round-trip constant and never crosses 1×.

This study asks the positive question: **are there whole-circuit families whose
`full_reduce` the database does faster than pyzx?** Answer: **yes — wide,
shallow, sparsely-but-non-locally connected gadget-rich circuits, with QAOA as
the archetype.** The DB is up to **2.3× faster** there, and the margin grows
with width.

## Method

For each family we sweep its natural size dimension and time `full_reduce` to a
fixpoint on both sides (DB round-trips included in the DB time; one-time graph
load excluded, as in all other benchmarks — this isolates the reduction
engine). We record input/output vertex counts, DB round-trips, and the ratio.
Correctness of the DB reduction is verified at the tensor level (vs pyzx **and**
vs the original circuit) on the smallest instance of every family; the QAOA
winner is verified separately across n=3–6, p=1–2 (all `tensor: db==pyzx`), so
every reported win is a win on a **correct** reduction.

Numbers are from one machine (Memgraph 3.6 in Docker, pyzx 0.10.3); the
crossover *regime* is the portable result, not the absolute times.

## Results by family

| family | structure | sweep | crossover? | DB vs pyzx |
|--------|-----------|-------|-----------|------------|
| **qaoa** (shallow, wide) | 3-regular graph, p=2–3 layers, γ=π/4 | width n | **yes, n≈24** | **0.44–0.6× (DB up to 2.3× faster)** |
| clifford_wide | genuine Clifford, depth 8n | width n | no | 1.4–1.8× |
| near_clifford | 10% T, depth 8n | width n | no | 2–6× |
| t_heavy | 50% T, depth 8n | width n | no | 4–6× |
| clifford_deep | 6 qubits, deep | depth | no | 5–8× |
| qaoa_dense | ~n/2-regular, p=2 | width n | no | 2.1–2.4× |

The winning regime, swept on width at fixed shallow depth (p=2, 3-regular):

| qubits n | input v | output v | DB | pyzx | ratio |
|----------|---------|----------|----|------|-------|
| 32 | 630  | 254 | 1.37 s | 2.43 s  | **0.56×** |
| 40 | 780  | 314 | 3.23 s | 5.45 s  | 0.59× |
| 48 | 960  | 386 | 5.15 s | 8.65 s  | 0.60× |
| 56 | 1110 | 446 | 5.87 s | 13.44 s | **0.44×** |

The DB wins from n≈24–32 and the ratio *improves* with width (pyzx grows faster),
reaching 2.3× at n=56.

## Why QAOA wins where Clifford and T-heavy random circuits do not

This is the crux, and it is **not** "QAOA is non-Clifford" — `t_heavy` is far
more non-Clifford and loses badly (4–6×). Three properties have to hold together:

1. **Gadget-dominated reduction.** The non-Clifford ZZ rotations (γ=π/4) become
   phase gadgets. pyzx's gadget machinery — `gadget_simp` (matching gadgets to
   fuse) and `pivot_gadget` (extracting a gadget through a pivot) — is one of its
   more expensive per-rewrite operations, done in a pure-Python loop. The DB does
   each as a flat Cypher query.
2. **Non-local connectivity.** QAOA's random 3-regular interaction graph creates
   gadgets whose targets span the whole diagram. After reduction, pyzx must do
   **global** gadget/pivot_gadget work whose per-rewrite cost grows with width.
   `t_heavy`/`clifford` random circuits are **linear-nearest-neighbour** (gates on
   adjacent qubit lines), so their gadgets stay local and pyzx clears them
   cheaply — pyzx full_reduce of a wide local circuit finishes in milliseconds.
3. **Bounded treewidth (wide + shallow + sparse).** The DB's per-query cost is a
   scan whose width grows with the diagram's connectivity. Wide-but-shallow,
   3-regular keeps the intermediate graph sparse, so each DB query stays cheap
   while the **number of independent gadgets grows with width** — exactly the many
   cheap-for-DB / expensive-for-pyzx operations that favour the DB. The round-trip
   count grows **sub-linearly** in width (≈117→371 over n=8→48) because each query
   batches a layer's worth of independent rewrites.

In short: QAOA maximizes the count of gadget rewrites pyzx must grind through
one-by-one, while keeping each DB query cheap. That is the only regime found
where the DB's structural disadvantage (round-trips, re-scans) is outweighed.

## The boundary of the winning regime

The win is fragile in two directions, both of which raise **treewidth** and blow
up the DB's per-query scans:

- **Depth.** At fixed width n=28, increasing layers reverses the win:
  p=2 → 0.87× (DB), p=3 → 1.12×, p=6 → 1.53× (pyzx). More layers entangle the
  diagram, raising treewidth.
- **Density.** `qaoa_dense` (≈n/2-regular) never crosses (2.1–2.4×) and the DB
  time explodes (13.6 s at n=24 vs 5.7 s pyzx) — dense connectivity = high
  treewidth = expensive DB scans and overlapping (non-batchable) rewrites.

So the sweet spot is a **ridge**: wide, shallow, sparse. Push width up → DB wins
more; push depth or degree up → DB loses.

## Practical takeaway — which real diagrams cross over

The database `full_reduce` is faster than pyzx on diagrams that are
**wide, shallow, sparse, and phase-gadget-rich with non-local interactions**:

- **QAOA / variational ansätze** on sparse (bounded-degree) interaction graphs,
  at shallow depth and many qubits — the demonstrated case.
- More generally, **Hamiltonian-simulation / phase-polynomial circuits** for a
  sparse, non-local Hamiltonian (e.g. a bounded-degree expander coupling), one or
  two Trotter layers, many qubits: the same gadget-heavy / sparse / wide profile.

It is **not** faster on the typical full-reduce workload — deep random circuits,
deep Clifford, T-heavy local circuits, or dense/all-to-all interactions — where
pyzx's in-process incremental matching wins. The crossover is a property of a
specific diagram *shape*, not of `full_reduce` in general.
