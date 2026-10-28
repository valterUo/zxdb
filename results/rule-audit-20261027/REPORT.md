# Rule audit — 2026-10-27

The final live regression run passed **91 tests in 85.946 seconds, with no
failures, errors or skips**. See [regressions.log](regressions.log). This is a
correctness audit, not a new performance campaign or a proof for every possible
input graph.

## Findings and fixes

| Finding | Correction and regression |
|---|---|
| Both Cypher supplementarity variants accepted non-graph-like neighborhoods. | Reject simple incident wires, non-Z neighbors, self-loops and parallel wires. Test adjacent/nonadjacent pairs in both variants, asserting no mutation for invalid candidates. |
| Gadget fusion grouped equal vertex neighborhoods without requiring valid target wires. This changed the tensor. | Require the color-aware parity-gadget wire pattern and unique targets. Test valid fusion and mismatched target wires in control and optimized paths. |
| Spider fusion lost the pi contribution of a Hadamard self-loop; a simple self-loop could prevent a valid fusion. | Normalize Z/X self-loops before pair matching, retaining the Hadamard phase contribution. Test both colors and both loop types with independent tensors and progress assertions. |
| Identity removal could stop before normalization exposed a degree-two identity. | Normalize input loops/parallel wires before the first removal pass. Test Z and X identities hidden by a simple self-loop in both variants. |
| A single pi-axle gadget was normalized but the method returned zero progress. | Include axle normalization in the returned count. Test a lone gadget, semantic preservation, and a zero-count second invocation. |
| Retained bialgebra clique queries discarded nonzero phases. | Require phase-zero clique vertices and the complete degree constraint, and give replacement vertices explicit zero phases. Test valid K2,2 and invalid phased/wire variants. |
| The retained Python gadget procedure ignored its graph_id argument and rewrote another diagram. | Scope candidate selection; validate ownership, loops, duplicate wires and edge types before mutation. No-argument calls require one owned graph. Test foreign-graph preservation and rejection of crossing edges. |
| The Python gadget procedure deleted an existing simple neighbor edge when toggling Hadamard adjacency. | Preserve that edge and add pi, matching the typed edge algebra. A dedicated tensor counterexample now passes. |

The first-run logs retain counterexamples:
[supplementarity](rule-audit-before.log),
[gadgets and loops](rule-audit-extra-before.log), and
[Python gadget semantics/isolation](rule-audit-python-before.log).
The first log also contains seven initial test-dispatch errors: Hadamard
cancellation has no optimized wrapper, so the final test correctly runs that
family through its implemented control entry point only.

## Coverage

All eleven implemented public rule families were exercised: spider fusion,
identity removal, Hadamard cancellation, local complementation, ordinary pivot,
gadget pivot, boundary pivot, phase-gadget fusion, bialgebra, copy, and
supplementarity.

The new audit transforms all 108 existing rule fixtures by reversing vertex
allocation and edge direction, conjugating phases, reversing ordered ports, and
changing boundary wire types. These produce 209 control/optimized live cases
(Hadamard cancellation has only its control implementation). Each output is
compared with its own input using PyZX's dense tensor contraction, independently
of the evaluation pipeline's quimb oracle. Named invalid cases must remain
unchanged; fixed-point rules must report zero and leave the graph unchanged on
the second call. Hadamard cancellation intentionally exposes a single labeling
pass and is excluded from that fixed-point assertion.

Additional matrices cover both copy colors, both Pauli phases, mixed incident
wires, existing edges between neighbors, both bialgebra-center phases, invalid
supplementarity neighborhoods, self-loops, and legacy query behavior. Offline
checks validate the transformed reference fixtures and the copy/bialgebra
matrices without claiming to execute database code.

Native tests add 49 transformed contexts across the five scoped rule families
and nine contexts for the legacy C++ gadget wrapper. The retained Python gadget
module is also executed live. Existing tests continue to cover all six scoped
C++ procedures, boundary/pivot neighborhood combinations, overlapping batches,
graph isolation, zero factors, full-reduction circuits, tensor-oracle behavior,
and evidence serialization.

## Reproduction

The run used the repository's Python environment and a fresh, dedicated
`memgraph/memgraph:3.14.0` container named `zxdb-rule-audit`, bound only to
`127.0.0.1:7691`. Existing databases were not used. The C++ source was compiled
inside that container with `g++ -O3 -std=c++20 -fPIC -shared`, using its SDK and
the matching missing header retained in the earlier validation container.
The Python module was copied into the container's query-module directory.

After deploying the modules and starting an empty dedicated database:

```powershell
$env:ZXDB_LIVE = '1'
$env:ZXDB_PYTHON_MODULE = '1'
$env:ZXDB_URI = 'bolt://127.0.0.1:7691'
myenv/Scripts/python.exe -m unittest discover -s tests -v
```

The regular evaluation runner now includes `test_rule_audit`. It includes the
optional Python-module tests when `ZXDB_PYTHON_MODULE=1`; compiled-native tests
remain enabled by its native variant. See [environment.json](environment.json)
for package versions and source hashes.

## Limits

Equivalence is up to a **nonzero global scalar**, with ordered input/output
ports preserved. General scalar tracking remains incomplete. Small dense tests
explicitly distinguish zero and nonzero results, but this numerical fixture
suite is not a symbolic proof for arbitrary phases or graph sizes.

Legacy JSON import/export limitations and the missing query referenced by
`turn_hadamard_gates_into_edges` remain outside the implemented rewrite-family
checks. The latter is an unavailable conversion API, distinct from the tested
Hadamard cancellation rule. Analytical-mode concurrency/crash safety and
performance were not established by this audit. Historical benchmark results
describe their original sources; these correctness fixes require new timing
runs before making performance claims about this revision.
