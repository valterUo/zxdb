# ZXdb code and system audit — 2026-09-08

## Follow-up: graph isolation implemented

Finding 1 is addressed for the Cypher paths: all 31 query entries now scope
their starting matches and marker cleanup to `$graph_id`, and the Python guards
and Hadamard labeling calls pass the selected ID. Python rewrite methods reject
relationships crossing graph ownership before mutation. Neighborhood matches
retain their original structure to preserve analytical-mode rewrite behavior.

The native procedure still has no graph parameter. Its Python wrappers now
reject mixed-graph databases before any rewrite; a missing requested graph is
a no-op. The importer and the other correctness findings below remain pending.

Validation on Memgraph 5.9.0: 107/107 rule cases, 112/112 full-reduce cases,
23/23 tests in full unittest discovery, and 9/9 tests in the final expanded
isolation suite. Isolation tests exercise all 107 rule fixtures with a second
diagram, all 31 query entries with a missing graph, marker cleanup, auxiliary
and composed reductions, native-wrapper refusal, and repeated boundary-pivot
tensor preservation. These totals overlap; the nine isolation tests include
eight already covered by the full discovery run.

Results: `evaluation/audit_results/rules_graph_id.log`,
`full_reduce_graph_id.log`, `unittest_graph_id.log`, and `graph_isolation.log`.
The initial audit below records the implementation as it was before this fix.

## Follow-up: typed Hopf cancellation implemented

Finding 2 is addressed. The `Hopf` query now cancels only simple wires between
opposite-colored spiders. The shared three-query normalizer handles the full
color-dual laws: same-color H pairs / opposite-color simple pairs cancel;
same-color simple duplicates / opposite-color H duplicates collapse; mixed
pairs keep the copying edge and add pi to one endpoint. Existing graph scoping
is retained.

The exact audit counterexample is now `sf_hadamard_parallel_via_fusion` in the
rule evaluation, full-reduce evaluation, and full-reduce unittest fixtures.
`tests/test_hopf_normalization.py` checks 60 color/multiplicity combinations,
16 fusion/composition variants, direct Hopf behavior, normalization stability,
and preservation of another diagram. Raw multigraph fixtures disable PyZX's
automatic edge simplification so the database must perform the rewrites.
Tensor checks use PyZX's dense contraction with scalars retained, avoiding the
known quimb-oracle issues. The pre-fix normalization queries produce 43 failing
subcases and no errors; the fixed queries pass all four targeted test methods.

Expanded evaluations pass 108/108 rule cases and 113/113 full-reduce cases.
Full unittest discovery passes 28/28 tests, including graph isolation and all
four Hopf normalization test methods with their parameterized subcases.
Evidence is in `evaluation/audit_results/hopf_before.log`, `hopf_after.log`,
`rules_hopf.log`, `full_reduce_hopf.log`, and `unittest_hopf.log`.
Other audit findings, including zero-scalar preservation, remain pending.

## Verdict and scope

**Do not sign off general correctness of `full_reduce` yet.** The current Cypher
pipeline executes successfully on the running Memgraph instance and passes its
existing evaluation fixtures, but new small examples demonstrate semantic
corruption and cross-graph mutation. The verification machinery also has false
positives that weaken existing correctness claims.

This is step 1 of the requested work: paper review, static review of the Python
driver and query collection, live baseline evaluations, and diagnostic
reproductions. Production rules and existing tests were not changed. Extending
the regression suite and repairing the findings are subsequent work. The native
module implementation has not received its detailed step-3 audit.

The paper is `2511.13033v1.pdf`, *ZX-DB: A Graph Database for Quantum Circuit
Simplification and Rewriting via the ZX-Calculus*. Sections II–III and Figures
4–15 establish the relevant rule semantics and the danger of interacting
matches. Its principal systems distinction is independent versus dependent
rewrites: selecting compatible matches can dominate execution. The current
implementation has evolved beyond the paper: many rules now use one match per
query, identity removal relies on analytical-mode visibility, and `full_reduce`
includes copy and supplementarity following PyZX 0.10.3.

## Environment and evidence

- Memgraph **5.9.0 Community**, reported by `CALL dbms.components()`; localhost
  Bolt port 7687. Python reference: repository `myenv`, PyZX **0.10.3**.
- Initially **zero nodes**, `IN_MEMORY_TRANSACTIONAL`. Constructing `ZXdb`
  switched the database to `IN_MEMORY_ANALYTICAL`. This is a global side effect.
- Node property indexes on `t`, `graph_id`, `id` and Wire property indexes on
  `t`, `id` were confirmed present after evaluation.
- Rule evaluation: **107/107**, all reported `tensor:db==pyzx`.
- Full-reduce evaluation: **112/112**, 111 tensor comparisons and one sampled
  tensor comparison; approximately 15 seconds of reported evaluation time.
- Additional comparison of all 107 rule fixtures against their original
  diagrams: **107 true**, using the existing quimb checker. This shares that
  checker's limitations described below and is not an independent oracle.
- Diagnostic probes reproduce the failures below. All probe-owned nodes were
  removed. Database is empty at completion; its mode remains analytical.
- No crash, recovery, concurrency, or large-scale soak certification was done.
  Historical SIGSEGV claims in comments were not reproduced in this audit.

Reproduce from the repository root using `myenv/Scripts/python.exe`:

```text
python -m evaluation.run_eval
python -m evaluation.eval_full_reduce
python -m evaluation.audit_original_semantics
python -m evaluation.audit_system_probes
```

These are **disposable-database commands**: the baseline harness clears the
entire database when importing each fixture. The two new diagnostic scripts
refuse to start with nonempty databases. Run them sequentially; clear only the
known evaluation fixture before starting the next diagnostic.

Evidence: [system probes](evaluation/audit_results/system_probes.json),
[original comparisons](evaluation/audit_results/original_semantics.json),
[rule log](evaluation/audit_results/rules_baseline.log), and
[full-reduce log](evaluation/audit_results/full_reduce_baseline.log).
Logs are ignored by the repository's existing `*.log` rule.

## Findings, ordered by impact

### 1. P1 — Every rewrite ignores the graph identifier

**Locations:** `zxdb/main_queries.json` (all 31 entries),
`zxdb/zxdb.py:111`, `:631`, `:915`, `:1000`.

None of the 31 query strings references `$graph_id`. Passing the parameter from
Python does not scope a query. The inline existence guards are likewise global.
Consequently `full_reduce('nonexistent')` reduced a different graph from five
vertices to three. Individual rules, normalization, color change and isolated
cleanup can all alter unrelated diagrams. Bialgebra's global side markers also
allow unrelated unfinished applications to be connected together.

Scope all matching and marker operations to the selected graph, validate that
wires do not cross graph ownership, and add two-graph preservation tests for
every public mutation. The `Node(graph_id)` index currently cannot help the
unfiltered rewrite scans.

### 2. P1 — Hopf cancels the wrong edge types and corrupts full_reduce

**Locations:** `zxdb/main_queries.json:23` (Hopf), `zxdb/zxdb.py:674`.

Hopf collects **all** wires between Z and X spiders and keeps their parity,
without testing `w.t`. The parity cancellation applies to simple Z–X wires;
parallel Hadamard Z–X wires obey the color-dual same-color simple-wire law.

Reproduction: an input boundary connects to Z(1/4) `a`; `a` connects simply to
Z(1/4) `b`; both connect by H wires to X(1/4) `x`; `x` connects to an output.
Fusion of `a,b` produces two H wires to `x`. Hopf deletes both, disconnecting
the input and output components. Both `spider_fusion` and `full_reduce` then
fail tensor equivalence to the original. The input has neither parallel edges
nor self-loops: this is not merely malformed database input.

Implement the complete color/edge normalization algebra; restricting this
query to simple wires is necessary but leaves opposite-color H duplicates to
be handled. Add the concrete fixture before changing the implementation.

### 3. P1 — Import silently does nothing by default, or clears other data

**Location:** `zxdb/zxdb.py:383`–`:501`.

Vertex and edge creation are indented inside `if initialize_empty`. The default
`initialize_empty=False` therefore imports zero vertices without an error.
Setting it true executes unrestricted `MATCH (v) DETACH DELETE v`, including
non-ZX data. The log misleadingly says it cleared the selected graph.

Both behaviors were reproduced with a two-boundary wire. Move creation outside
the conditional, make replacement graph-specific, and keep whole-database
wiping an explicit separate API. Define duplicate-import semantics and enforce
node identity within a graph.

### 4. P1 — Zero maps become nonzero; scalars are not preserved

**Locations:** `zxdb/main_queries.json:391`, `zxdb/zxdb.py:193`, `:358`.

An identity wire plus an isolated Z(pi) vertex is the zero map. `full_reduce`
deletes the isolated vertex and exports the identity map. PyZX's direct tensor
contraction confirms zero input and nonzero output. An input graph with
`g.scalar.is_zero=True` also exports with `is_zero=False`, even without a
rewrite. The importer saves scalar information only in a sidecar; export never
restores it, and rewriting never updates it.

Ignoring a **nonzero** global scalar can be a documented contract; discarding a
zero scalar changes the map and cannot be justified that way. Store graph-level
scalar information, at minimum zero status with sound propagation, or reject
unsupported scalar cases explicitly. Removing closed components also needs
scalar-aware semantics, not unconditional deletion.

### 5. P1 — The correctness checker hides scalar errors

**Locations:** `zxdb/pyzx_utils.py:276`, `evaluation/harness.py:115`, `:347`,
`:438`.

- For a degree-zero Z spider, the tensor builder assigns both terms to the
  same scalar array element, overwriting 1 with `exp(i*phase)` instead of
  adding them. It evaluates Z(pi) as -1 rather than 0.
- The quimb tensor network does not include `g.scalar`; `_clean_copy` also
  loses that field. Explicit zero scalars are invisible.
- Both zero-map failures above are consequently reported equivalent by the
  quimb checker. Its arrays are preserved in the probe results; the direct PyZX
  check of the isolated-spider example disproves that verdict.
- `_verdict` returns success for `unverified` and degree-sequence agreement.
  Degree sequences do not establish semantic equivalence.
- DB/PyZX agreement suppresses the original-input comparison. Shared incorrect
  behavior can pass. Non-confluence explains different **graphs**, not unequal
  tensors of two sound rewrites of the same input.
- Out-of-range phases are measured but omitted from `run_case`'s success gate.

Repair and independently validate the oracle before relying on larger green
test totals. Separate verified, failed, and inconclusive results; compare to
the original wherever feasible and treat proven mismatches as failures.

### 6. P1 — Standalone supplementarity lacks graph-like preconditions

**Location:** `zxdb/main_queries.json:231`, `:247`.

Both queries compare neighbor sets without validating incident edge types or
the required neighbor types. A valid ZX diagram with two Z leaves (phases 1/4
and 3/4) attached by simple wires to the same two Z targets triggers deletion
and phase propagation, but the resulting tensor differs. This was reproduced
with input/output boundaries attached to the two targets.

This finding concerns the public standalone rule. It does not demonstrate
failure of this particular rule inside an already graph-like pipeline. Add
explicit guards or a validated input-domain contract. Gadget fusion and the
pivot edge toggles also assume normalized neighborhoods; audit these domains
with wrong-type and multiedge negative fixtures instead of assuming that
PyZX's preconditions follow automatically for every public method.

### 7. P2 — Export silently changes multigraphs

**Location:** `zxdb/zxdb.py:268`.

Export drops self-loops and keeps only the first relationship for an unordered
endpoint pair, irrespective of edge type. A simple-plus-H pair exported as one
edge in the probe. Selection is not ordered, so retained type is not a stable
contract. Unsupported edge types are also coerced to simple edges. Export
must preserve multigraph semantics or explicitly reject invalid states.

Spring-layout coordinates can additionally make PyZX's naive `to_tensor`
raise `ValueError: axes don't match array`; this occurred during the probes.
Keep semantic export independent of visualization layout. Validate boundary
degree/order and supported vertex types before reduction. Unlike PyZX's
`full_reduce`, this implementation does not reject H-box input up front.

### 8. P2 — Hadamard conversion references a missing query

**Location:** `zxdb/zxdb.py:1201`.

`turn_hadamard_gates_into_edges()` accesses `"Turn Hadamard gates into edges"`,
which is absent from the query collection. Calling it raises `KeyError`.
`hadamard_edges=True` invokes this after import, so it can fail after data has
already been loaded. Implement and validate the conversion, or remove/reject
the unsupported option before mutations begin.

### 9. P2 — Mutation is neither atomic nor comprehensively bounded

**Locations:** `zxdb/zxdb.py:26`, `:85`, `:915`, `:983`.

`_execute` uses a session, so each query is autocommitted; it does not use one
transaction per rule as the `full_reduce` docstring says. Color change and
bialgebra each span multiple statements. Hadamard labeling leaves intermediate
markers if interrupted. Analytical mode has no normal rollback protection.
An error can leave a partially rewritten diagram, and reconnecting is not
recovery. The constructor globally switches storage mode and swallows every
index-creation exception, masking operational failures.

`max_rounds` limits only the outer loop. Inner rule fixpoints and Clifford loops
are unbounded, and the initial reductions run before the cap. It is not an
overall timeout or rewrite budget. Add shared budgets/cancellation and validate
arguments before mutation. Do not assume that switching to transactional mode
alone is sound: batched queries require revalidation of visibility semantics.

Memgraph documents the analytical-mode tradeoff in
[its storage-mode explanation](https://memgraph.com/blog/memgraph-storage-modes-explained).
Successful autocommit runs here do not prove crash safety or support the
comments' claim that autocommit eliminates every deleted-node hazard.

### 10. P2 — Reproducibility and advertised query coverage are incomplete

**Locations:** `requirements.txt`, `README.md`, `zxdb/zxdb.py:55`,
`zxdb/main_queries.json`.

Dependencies are unpinned despite version-sensitive PyZX behavior. Query loading
is relative to the working directory, so construction fails outside the repo
root. README references nonexistent `query_collections` and `test` directories.
The JSON mixes live queries, old bialgebra implementations, three `original`
queries, and experimental batched pivot. Baseline success validates the active
driver paths, not every legacy query advertised in that collection. Split or
label these and validate query names at startup.

## Rule-by-rule assessment

Every row inherits the graph-isolation and scalar limitations above. Passing
fixtures establish executable Cypher and agreement on those fixtures, not a
proof over arbitrary diagrams.

| Rule | Existing cases passing | Assessment |
|---|---:|---|
| Spider fusion + Hopf | 11 | Blocking typed-edge counterexample; batch rewiring depends on visibility |
| Identity removal + parallel normalization | 11 | Covered edge combinations pass; batched chain behavior mode-dependent |
| Hadamard cancellation | 7 | Covered paths pass; bounded-length path search, unscoped markers, created wire lacks graph_id |
| Local complementation | 11 | +/- pi/2 and interior/H-neighbor guards present; assumes normalized input |
| Pivot (interior and Pauli boundary) | 12 | Phase transfer and three-group toggles match paper; one selected match per query |
| Pivot gadget | 9 | Non-Pauli, degree and interior guards present; one match per query |
| Pivot boundary | 7 | Boundary and neighbor guards present; assumes single normalized boundary wire |
| Phase gadget fusion | 9 | Signed phases and axis normalization covered; normalization count excluded from return |
| Bialgebra | 10 | Active PyZX expansion path covered; two statements with global temporary side markers |
| Supplementarity | 10 | Covered graph-like fixtures pass; standalone simple-edge counterexample fails |
| Copy | 10 | Color/edge compatibility guards present; covered propagation cases pass |

Color change and isolated cleanup are exercised through composition, not a
dedicated API-level fixture matrix. `full_reduce`'s rule order matches installed
PyZX 0.10.3, but lacks its H-box rejection and scalar-aware cleanup. It does not
call standalone bialgebra or Hadamard cancellation, consistent with that PyZX
pipeline. Experimental batched-pivot correctness is not established by these
default-path evaluations.

## Follow-up order

1. Fix the oracle and convert the reproduced failures into strict regression
   tests; preserve evidence from the original implementation.
2. Repair graph isolation, importer behavior, typed Hopf normalization, zero
   semantics, and rule-domain validation. Then verify input/output round trips,
   repeated reductions, two coexisting diagrams, and negative fixtures.
3. Audit transactional/error behavior and add reproducible dependency/server
   setup. Rerun all rule, composition and unittest suites after repairs.
4. Review `zxdb_qm/zxqmcpp.cpp` and `zxqm.py` in detail. Both procedures are
   currently registered in Memgraph. The native path replaces only pivot-gadget
   reduction, not the full Cypher pipeline. Its README attributes the speedup
   to greedy compatible matching in process, avoiding one-match rescans and
   round trips. This is a historical explanation to verify with profiling in
   step 3, not a newly measured performance conclusion.
