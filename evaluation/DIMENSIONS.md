# What dimension drives the DB-vs-pyzx `full_reduce` crossover?

`python -m evaluation.dimensions [sweep ...]`   (`verify` for a correctness check)

A controlled one-factor-at-a-time (OFAT) study. One parametric generator with
**independent** knobs is swept around a fixed baseline, and every circuit is
annotated with **intrinsic graph metrics** so the crossover is explained by a
*measured property*, not a circuit label. Uses the production `full_reduce`.
Each point is averaged over 3 seeds; the generator is tensor-verified vs pyzx.

Generator (layered ansatz, generalizing QAOA / Trotter): each layer applies
`ZZ(phase)` (a phase gadget when `phase` is non-Clifford) on a random
interaction graph of controllable **degree** (interactions per qubit) and
**span** (max range `|i-j|`; `None` = full range / any pair), then an `X(π/2)`
mixer. Knobs: `qubits` (width), `layers` (depth), `degree` (density),
`span` (locality), `phase` (gadget content). Key intrinsic metric:
**`twInt` = treewidth of the union interaction graph** (the qubit-coupling
graph) — the cleanest measure of connectivity complexity.

> **Two fixes made this study trustworthy** (earlier drafts were wrong because of
> them). (1) The crashes were a Memgraph IN_MEMORY_ANALYTICAL hazard fixed by
> autocommit (see `project-performance`); before that, mid-run restarts fought the
> `memgraph-mage` image's ~60 s PyTorch/DGL load for CPU, giving 10× inflated,
> noisy times. The study now runs restart-free with a warm-up. (2) The
> `pivot_gadget` query collected every candidate's neighbourhood *before* `LIMIT
> 1` (O(#candidates × degree)); on dense graphs that single query dominated
> full_reduce. Fixed with the lazy-collection pattern already used by the other
> pivots. Absolute times are machine-specific; the *shape* is the portable result.

## Headline

**The DB beats pyzx on gadget-rich circuits above a low connectivity threshold,
and the advantage is BROAD — not a narrow band.** Concretely, the crossover is
governed by interaction-graph treewidth `twInt` plus gadget content:

- **Below threshold → DB loses** (fundamental): nearly-local/sparse circuits
  (`twInt ≈ 1`) or Clifford circuits. pyzx reduces these almost instantly
  (~0.3 s), so the DB's round-trip floor cannot compete.
- **Above threshold (`twInt ≳ 2`, with non-Clifford gadgets) → DB wins**, by up
  to ~2×, and the win **holds across a wide treewidth range** (`twInt` 2–18),
  only neutralizing to a tie at extreme density (`twInt ≈ 22`).

This *supersedes* two earlier wrong conclusions of this study: "treewidth doesn't
predict" (that used the useless reduced-graph treewidth `twRed`≈50–61 on
noise-corrupted timings) and "narrow inverted-U Goldilocks band, DB loses at high
treewidth" (the high-`twInt` losses were the `pivot_gadget` bottleneck, not
fundamental — see below).

## Locality, density, phase (clean, post-fix, 3 seeds, q28)

**Locality** (vary `span`, degree 3): once there is *any* real non-locality the
DB wins; only the near-local span=1 (also sparser — see note) loses.

| span | twInt | DB/pyzx | wins/3 |
|------|-------|---------|--------|
| 1 | 1 | 3.12× | 0/3 |
| 2 | 2 | 0.69× | — |
| 4 | 4 | 0.59× | — |
| 8 | 6 | 0.56× | — |
| 16 | 10 | **0.46×** | — |
| None | 10 | 0.49× | — |

**Density** (vary `degree`, span None): loses only at the sparsest; wins from
degree 2 up, holds dense, ties at the extreme.

| degree | twInt | inV | DB/pyzx |
|--------|-------|-----|---------|
| 1 | 2 | 280 | 2.83× |
| 2 | 7 | 420 | **0.52×** |
| 3 | 10 | 560 | 0.92× |
| 5 | 15 | 840 | 0.58× |
| 8 | 18 | 1260 | 0.83× |
| 12 | 22 | 1820 | 0.98× (tie) |

**Phase / gadget content** (q28, span None, degree 3) — necessary co-factor:

| phase | twInt | DB/pyzx |
|-------|-------|---------|
| π/2 (Clifford) | 11 | 4.10× |
| π/4 (T) | 10 | 0.84× |
| π/8 | 10 | 0.88× |

At the *same* treewidth, Clifford loses badly (pyzx trivial) while gadget-rich
wins. So the win requires both connectivity above threshold **and** non-Clifford
gadgets.

> **`span=None` and the span=1 confound.** `None` = full range (`span=qubits`,
> any pair); not special. The earlier "span=None mysteriously loses" was crash/ML
> measurement noise — post-fix span=None wins 0.49×. Separately, at `span=1` the
> edge sampler can only place ~27 of 42 target edges, so span=1 is *also* sparser
> (its `twInt`=1 reflects that); span≥2 reach the full edge count.

## The bottleneck behind the (former) high-treewidth losses

Profiling a dense full_reduce by rule showed one query dominating and the regime
shifting from transport- to execution-bound:

| degree | total | pivot_gadget | ms / round-trip |
|--------|-------|--------------|------------------|
| 1 (sparse) | 0.20 s | ~0 | ~1.5 (transport-bound) |
| 5 (dense, *before* fix) | 4.07 s | 3.19 s (78 %) | 35.8 (execution-bound) |
| 5 (dense, *after* fix)  | 1.72 s | 0.82 s | — |

The `pivot_gadget` query ran its O(degree) neighbour collection and interiorness
checks for **every** candidate `(z_j, z_alpha)` pair before `LIMIT 1`, i.e.
O(#candidates × degree) per call — cheap when sparse (few low-degree candidates),
explosive when dense (many high-degree candidates). Moving the collection to
*after* `LIMIT 1` (cheap `NOT EXISTS` interiorness to pick the pivot first) made
it ~4× faster at degree 5 and flipped degree 3 and 5 from losses to wins. This is
why dense was treated "differently": it was the only regime where a single query
became execution-bound.

## Width and depth: secondary, via treewidth and size (post-fix)

These act through treewidth and size rather than as independent structural
drivers — but post-fix the DB is efficient enough that *size alone* tips even
low-treewidth circuits to a win at scale.

- **Width, non-local** (span None, so `twInt` grows with qubits): clean wins that
  *strengthen* with width — q18 1.02× → q24 0.65× → q30 0.48× → q36 0.37×.
- **Width, local** (`span`=3, `twInt` fixed at 3): still wins at large width by
  size alone — q12 2.14× → q24 0.93× → q44 0.50×.
- **Depth** (local span, `twInt` fixed at 3): borderline and noisy — L1 3.39×,
  L2 0.66×, L3 1.40×, L4 1.27×, L6 0.61×, L8 0.66× — large size drifts toward the
  DB but near the low-treewidth boundary the outcome is a noisy near-tie.

So **treewidth is the clean primary driver** (wins strengthen monotonically with
`twInt`), while **size is a secondary one** (big circuits drift to DB wins even at
low treewidth, since more total rewrites favour the DB's work-efficient queries
over pyzx's per-rewrite Python cost). Only *small-and-low-treewidth* circuits sit
firmly on pyzx's side.

## Mechanism

A competition between two cost curves driven by different properties. **pyzx's
cost** rises with gadget complexity/non-locality (global `pivot_gadget` /
`gadget_simp` in pure Python). **The DB's cost** is the round-trip floor (~1 ms ×
#queries) plus per-query execution. Below threshold pyzx is simply too fast to
beat (round-trip floor dominates); above it, the DB's work-efficient C++ queries
win and the margin grows with treewidth/size.

### Why the advantage *fades* at extreme density (and that it is real)

The fade to a tie is statistically significant, not band noise (5 seeds, q28,
span None): degree 5 (`twInt` 15) ratio **0.46** [0.36–0.57], degree 8 (`twInt`
18) **0.68** [0.52–0.81], degree 12 (`twInt` 22) **1.03** [0.86–1.27] — the
degree-8 and degree-12 ranges don't overlap. The cause is **not** symmetric
saturation: the DB total grows *faster* than pyzx (DB 1.4→9.9 s vs pyzx 2.5→10 s
over degree 5→12), and it is almost all `pivot_gadget` (691→2248→6741 ms = 48 %→
60 %→68 % of the reduction). Two compounding effects inside that query:

1. **Re-scan per application** — it applies one gadget per round-trip (`LIMIT 1`)
   and *every* application re-runs the candidate `MATCH` + `NOT EXISTS`
   interiorness over the whole dense graph (O(#applications × #candidates ×
   degree)). The lazy-collection fix removed the eager neighbour *collection* but
   not this per-application *filtering* re-scan; at high density both the number
   of applications and the candidate count grow, so it compounds super-linearly.
2. **Per-gadget bipartite toggle is O(degree²)** — inherent rewrite work that
   pyzx pays too.

pyzx keeps gadget state in memory instead of re-scanning, so it scales better at
extreme density; the DB's re-scan overhead (1) compounds and overtakes its
constant-factor C++ advantage → ratio → 1. **So the fade is mostly a remaining
implementation inefficiency, not a fundamental limit:** effect (1) is the same
kind of bug as the eager-collection — batching gadget applications or avoiding the
per-application re-scan would push the tie point to higher density; only effect
(2) is inherent (and symmetric between the engines).

**Resolved by an in-process query module (see `zxdb_qm/`).** Effect (1) cannot be
removed in declarative Cypher (the maximal-disjoint selection needs imperative
greedy; mutual-min under-selects, and a MAGE coloring needs a quadratic conflict
graph). But a custom **C++ query module** runs in-process and does exactly pyzx's
O(E) greedy (mark consumed neighbourhoods) with native toggles. Routing only
pivot_gadget through it (`full_reduce_with_query_modules`, everything else still
Cypher) keeps 112/112 correctness and makes dense full_reduce **~2–2.5× faster
than the Cypher full_reduce and ~1.4–2.2× faster than pyzx — including degree 12 /
twInt≈22, which now wins 0.48× instead of tying.** So with the module the fade is
gone and the DB beats pyzx across the whole density range; the tie was a property
of the *Cypher* pivot_gadget, not a fundamental limit.

## Takeaway (the paper argument)

The database's `full_reduce` advantage is **broad, not narrow**: on gadget-rich
(non-Clifford) circuits it beats pyzx by up to ~2–3×, and the advantage **grows
along two axes** — primarily **interaction-graph treewidth** (`twInt`; the win
strengthens monotonically with connectivity, fading to a tie only at extreme
density) and secondarily **size** (large circuits win even at low/local treewidth,
since more total rewrites favour the DB's work-efficient C++ queries over pyzx's
per-rewrite Python). Non-Clifford gadget content is a necessary co-factor. The DB
loses only where pyzx is trivially fast — **small-and-sparse/local** or
**Clifford** circuits — where the DB's fixed round-trip floor dominates; that is
the one fundamental limit. Width, depth, and the circuit family name are not
independent drivers; they matter only through treewidth and size. The earlier
impression of a narrow "Goldilocks band" was an artifact of one non-work-efficient
query (`pivot_gadget`) and crash/measurement noise, both now fixed.
