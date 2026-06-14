# Scalability study: where does the database beat pyzx?

`python -m evaluation.scalability [rule ...]`

For each rewrite rule (and `full_reduce`) we find the **sweet spot** — the
graph size at which Memgraph's query engine evaluates the rule faster than
pyzx in wall-clock time. No correctness checking (speed only; correctness is
covered by `run_eval` / `eval_full_reduce`).

## Method

We time **rule execution only**: the graph is pre-loaded into the DB and
pre-built in pyzx, so the one-time load/build is excluded — this isolates
engine speed (Memgraph C++ vs pyzx Python). Both sides run the rule to a
fixpoint. For each rule we use a graph family `G(N)` whose work grows with `N`
while the number of DB round-trips stays roughly constant, so the comparison
is not dominated by network overhead:

- **Batched rules** (spider fusion, identity, hadamard) rewrite a maximal set
  of disjoint trigger patterns in one query → scale by the **number of
  disjoint patterns**.
- **Single-application rules** (lcomp, pivot, …) do one rewrite whose cost
  grows with the **degree** of the matched spider(s) (O(N) or O(N²) edges /
  spiders touched in one query) → scale by that degree.

Numbers below are from one machine (Memgraph 3.6 in Docker, pyzx 0.10.3); the
crossover *points* are the portable result, not the absolute times.

## Results

| rule | scale dim | crossover (DB wins from) | DB vs pyzx at largest N | verdict |
|------|-----------|--------------------------|--------------------------|---------|
| **identity** | # patterns | **200 patterns ≈ 600 v** | ~200× faster (5000 pat.) | ✅ big win |
| **pivot** | spider degree | **64 ≈ 130 v** | ~3× faster (deg 1024) | ✅ win |
| **copy** | neighbours | **512 ≈ 514 v** | ~2× faster (1024) | ✅ win |
| **spider_fusion** | # patterns | **200 ≈ 800 v** | ~95× faster (5000 pat.) | ✅ win (after O(E) matching) |
| **pivot_gadget** | spider degree | **512 ≈ 1026 v** | ~3× faster (deg 1024) | ✅ win |
| **pivot_boundary** | spider degree | **1024 ≈ 2051 v** | ~1.6× faster (deg 1024) | ✅ win (late) |
| gadget_fusion | # gadgets | none (≤10000) | ~2–5× slower | ✗ |
| hadamard | # patterns | none | 33–130× slower | ✗ |
| lcomp | spider degree | none | blows up (11.7 s @ deg 512) | ✗ |
| supplementarity | # shared nbrs | none | blows up (2.0 s @ 1024) | ✗ |
| bialgebra | spider degree | none | blows up (14 s @ deg 256) | ✗ |
| **full_reduce** | qubits | none ≤16 q (536 v) | 3.4× slower @ 14 q, ratio falling | ✗ in range |

### Why the winners win

The DB beats pyzx exactly when its Cypher query is **work-efficient** and the
per-rewrite cost in pyzx's pure-Python loop grows faster than the DB's C++
execution:

- **identity** is the standout: one batched query removes *all* degree-2
  phase-0 spiders, so DB time barely grows (0.006 s → 0.068 s from 1 to 5000
  patterns) while pyzx grows linearly (→ 15 s). ~200× faster at 5000.
- **pivot / pivot_gadget / pivot_boundary**: the DB applies one rewrite whose
  edge/spider creation is O(N) (gadget) or a single set-based bipartite toggle
  (pivot); its time stays nearly flat (e.g. pivot_gadget ~0.005 s even at
  degree 1024) while pyzx's Python edge bookkeeping grows.
- **spider_fusion**: the query fuses a maximal independent set of pairs per
  call, so a 20 000-vertex graph fuses in ~2 queries; crosses at 2000 v.
- **copy**: one query creates the N propagated spiders; DB time stays ~0.01 s.

### Why the losers lose

- **hadamard_cancel** is not batched: its labelling implementation marks **one
  Hadamard chain per round-trip** in a Python loop, so N patterns cost N
  round-trips — it is 33–130× slower and never catches up. (Fixable by batching
  the labelling query.)
- **lcomp** and **bialgebra** scale as O(N²) but their Cypher does it slowly:
  lcomp toggles the complete graph K_N with a per-pair `OPTIONAL MATCH` +
  `FOREACH`, and bialgebra builds the N×N bipartite edges with nested `UNWIND`s.
  pyzx does the same O(N²) with a Python dict and wins (11.7 s vs 0.45 s at
  lcomp degree 512). (Fixable with set-based edge construction.)
- **supplementarity** pays an expensive "same-neighbourhood pair" match in
  Cypher (≈ O(N²) on the shared neighbourhood) that pyzx does with hashed
  parity sets.
- **gadget_fusion** stays only ~2–5× slower (no blow-up) but the grouping /
  `collect` overhead keeps it behind pyzx's efficient gadget matcher.

### full_reduce — bottlenecks and the optimization pass

Profiling one full_reduce on a q8/325-vertex circuit (timing each distinct
query, forcing execution) gave the real hotspots — which differ from the
adversarial single-rule study above, because real circuits trigger small,
heavily-cascading rewrites rather than one huge one:

| query | calls | time |
|-------|-------|------|
| spider fusion | 16 | 179 ms |
| boundary pivot (`single`) | 40 | 105 ms |
| interior pivot (`two`)    | 40 | 103 ms |
| parallel-edge guard / lcomp / identity / hopf | … | ~60 ms |

So full_reduce is bound by **per-query graph scans across ~130 queries**, not
by transaction overhead. Two correctness-preserving optimizations were applied
(all 107 per-rule + 112 full_reduce cases still pass at the tensor level):

1. **pivot loop restructure** — the boundary-pivot query is an expensive full
   scan that rarely matches but was run after *every* interior pivot. Running
   interior pivots to a fixpoint first and only then probing for a boundary
   pivot (pivots are confluent, so order is irrelevant) removes dozens of
   no-op scans: pivot 173 → 130 ms.
2. **spider-fusion existence guard** — after `to_gh` every interior wire is
   Hadamard and pivot/lcomp create only Hadamard wires, so spider fusion finds
   nothing fusable on most calls yet still paid an ~11 ms scan. A cheap guard
   using the `:Wire(t)` edge index (~1 ms) gates the heavy fusion query; it is
   behaviour-identical (the fusion query already returns 0 in that case).
   spider fusion 154 → 123 ms.
3. **spider-fusion matching: O(E²) `reduce` → O(E) mutual-min handshake** — the
   old query chose the fusion matching with a `reduce` over an accumulating
   node list (`collections.contains` per edge → O(E²)). It is replaced by a
   single-pass aggregation: each node's smallest same-colour simple neighbour
   is found with one `min()`, and an edge is kept iff its endpoints are each
   other's minimum. That is always a valid vertex-disjoint matching and always
   selects ≥1 edge when any fusable edge exists (the globally smallest fusable
   endpoint and its minimum neighbour are mutual), so the loop still
   terminates. **Standalone, ~42× faster than the old query at 5000 disjoint
   pairs (7.4 s → 0.18 s); the spider_fusion crossover moves 500 → 200
   patterns and it becomes ~95× faster than pyzx at 5000 pairs.** In
   full_reduce it trims spider fusion 123 → 95 ms.
4. **both pivot queries: EXISTS interiorness + lazy neighbour collection** —
   PROFILE on the boundary pivot showed the neighbour-collection `OPTIONAL
   MATCH`es ran for *every* candidate spider pair (291 collections to apply one
   rewrite). Both the interior pivot ("two interior Pauli spiders") and the
   boundary pivot ("single interior Pauli spider") were rewritten to (a) test
   interiorness with a cheap `NOT EXISTS { MATCH (a)-[w:Wire]-(x) WHERE w.t<>2
   OR x.t<>1 }` predicate that the planner short-circuits, (b) `ORDER BY id …
   LIMIT 1` to pick the single pivot to apply, and only *then* (c) collect the
   exclusive/shared neighbour partitions for that one chosen pair. Collection
   hits dropped 291 → ~15. Behaviour-identical (same rewrite, same fixpoint).
   In full_reduce: interior pivot 103 → ~80 ms, boundary pivot folds in too;
   **pivot_rule total 130 → 108 ms**.

Net: q8 full_reduce **445 → 364 → 316 → 267 ms (~40 %)**; the q16 ratio
improves 3.9× → ~3.4× and DB times drop across the board.

Verified correct after each change: 107/107 per-rule + 112/112 full_reduce at
the tensor level, the legacy `test_spider_fusion`, and a battery of adversarial
spider-fusion structures (chains, stars, triangles, complete graphs, 4-cycles,
bridged pairs/stars, X-X fusion, mixed Z-X non-fusion).

Two larger structural changes were tried and **rejected** because they did not
help (and risk correctness/stability):

- **Batching pivots** (apply a maximal non-interacting set per query, like
  spider fusion): on real circuits, pivot neighbourhoods overlap heavily, so
  the disjoint selection finds few to batch yet pays an expensive in-query
  `reduce` over candidate neighbourhoods — net **slower** (173 → 2070 ms).
  Interacting pivots must be serialized regardless.
- **One shared transaction** for the whole reduction: measured **no speedup** —
  not transaction-begin/commit-bound. (2026-06-14 follow-up pinned down what it
  *is* bound by: **bolt transport per round-trip**. A trivial `RETURN 1` costs
  ~1.05 ms/round-trip while a real rule query on ~300 nodes adds only ~0.4 ms of
  execution — so full_reduce's ~134-330 `tx.run` calls are ~1 ms of irreducible
  transport each. The only lever is *fewer* round-trips, i.e. combining rewrites
  per query; a shared transaction keeps the same number of `tx.run` calls and so
  cannot help. Necessary-condition guards also cannot help — a guard is itself a
  round-trip, and the correctness-mandated *loose* triggers fire on nearly every
  reduced graph, so they add round-trips rather than removing them. A small
  exception that *does* help: the to_gh X-spider guard, which gates 2 round-trips
  on 1, since after the first color change no X-spider ever exists, q8 140->134.)

### Why the per-rule speedups do NOT transfer to full_reduce

This is the crux. The per-rule crossovers above come from graphs where **one
rule does a single LARGE rewrite** — e.g. one pivot on a degree-1024 spider
does O(N²) edge work in ONE query, where the DB's single C++ query beats pyzx's
Python loop. full_reduce never enters that regime: it decomposes into **many
SMALL rewrites** (≈150 pivots/fusions at q8, each on a low-degree spider). For a
small rewrite the DB's per-op C++ advantage is tiny, and the per-rewrite
round-trip cancels it, so the DB is ≈ pyzx *per rewrite* but pays one round-trip
each. The speedup is a property of *large single rewrites*, not of the
*composition of many small ones*.

### Scaling up does not cross over — it plateaus

Pushing full_reduce to large circuits (depth = 20·qubits), DB vs pyzx wall
time, **after all four query optimizations above** (the DB column is the new,
faster time):

| qubits | vertices | DB | pyzx | ratio |
|--------|----------|----|------|-------|
| 16 | 518  | 0.67 s | 0.15 s | 4.4× |
| 24 | 790  | 1.58 s | 0.51 s | 3.1× |
| 32 | 1087 | 2.70 s | 0.84 s | 3.2× |
| 40 | 1323 | 2.85 s | 2.01 s | **1.4×** |
| 44 | 1458 | 3.70 s | 2.26 s | 1.6× |
| 48 | 1612 | 6.47 s | 3.70 s | 1.8× |
| 56 | 1847 | 7.76 s | 4.55 s | 1.7× |

The optimizations cut the DB's constant factor (q40 DB time **3.47 → 2.85 s**,
ratio **1.9× → ~1.5×**), but the ratio still **plateaus around 1.5–1.8×** at
large sizes and does **not** cross 1× — the per-circuit wobble (q40 dipped to
1.4×) is variance, not a trend. This is the predicted outcome: DB and pyzx scale
at the **same asymptotic rate** — both are O(rewrites × graph) — so the DB
carries a roughly constant ~1.5–2× factor from the round-trip floor. The query
optimizations *shrink that constant* (every one of them is kept) but cannot
remove it; the gap is structural, not a "scale further" problem.

### The only way to make full_reduce beat pyzx

Remove the round-trip floor: run the whole fixpoint **inside Memgraph** as a
server-side query module (MAGE / `mgp`), so the ~300 client round-trips and the
repeated full re-scans collapse into one in-process execution. That keeps the
verified rewrite logic but pays the round-trip once. It is a sizable change
whose correctness must be re-validated against `run_eval` / `eval_full_reduce`,
and is the recommended next step.

(The correctness-preserving query optimizations above — pivot loop restructure,
spider-fusion guard — still cut full_reduce ~18% and are kept; 107/107 per-rule
and 112/112 full_reduce cases remain tensor-verified.)

## Takeaway

The database is **not** a faster drop-in for small diagrams (round-trip and
serialization overhead dominate below a few hundred vertices), but it
**overtakes pyzx on large diagrams for every rule with a work-efficient query**
— identity, the three pivots, spider fusion and copy all cross over between
~130 and ~2050 vertices, and identity/pivot become multiple times faster at
scale. The remaining rules (hadamard, lcomp, supplementarity, bialgebra,
gadget_fusion) lose because of *implementation* inefficiencies (non-batched
labelling or non-set-based O(N²) Cypher), not a fundamental limit — they are
the clear targets for the next optimization pass.
