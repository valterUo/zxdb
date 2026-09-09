# ZXdb: ZX-calculus on graph databases

This implementation provides the core simplification rules from PyZX implemented as openCypher queries:
* Spider fusion
* Identity removal
* Local complementation
* Gadget fusion
* Pivot rule
* Pivot gadget rule
* Pivot boundary rule
* Bialgebra simplification

## Diagram ownership

Cypher rewrites in `zxdb/main_queries.json` require the `$graph_id` parameter.
`ZXdb` passes it from each rule method, including `full_reduce(graph_id)`.
Matches and temporary-marker cleanup operate on the selected diagram; a missing
graph ID is a no-op. `get_degree_distribution` is also scoped to that diagram.

`Node.graph_id` defines ownership. Diagrams must not have relationships to nodes
outside their graph. Python rewrite methods reject such connections before
mutation, including connections to nodes with no graph ID. Existing wires
without a `graph_id` property remain supported when their endpoints belong to
the same diagram. This check is not a concurrency lock: do not mutate graph
ownership or insert cross-graph connections concurrently with a rewrite.

The experimental C++ procedure is still database-wide. Its Python wrappers
reject databases containing other diagrams or unowned nodes before dispatch;
native graph filtering remains pending. The separately audited import behavior
is also unchanged: `initialize_empty=True` clears the whole database, so do not
use that importer option to load a second diagram into an occupied database.

Run the isolation regressions with:

```sh
python -m unittest discover -s tests -p test_graph_isolation.py -v
```

These tests load private graph IDs directly and remove only their own fixtures.

## Parallel-edge normalization

The normalizer handles both spider colors before and after fusion:

| Wires between two spiders | Same color | Opposite colors |
|---|---|---|
| Two simple wires | Keep one simple | Cancel both |
| Two Hadamard wires | Cancel both | Keep one Hadamard |
| One of each | Keep simple; add pi to one endpoint | Keep Hadamard; add pi to one endpoint |

The standalone `Hopf` query cancels only simple-edge pairs between opposite
colors. The shared normalizer also handles duplicate Hadamard and mixed wires.
Regression tests include the audit's fusion counterexample and compare small
diagrams using PyZX's dense tensor contraction:

```sh
python -m unittest discover -s tests -p test_hopf_normalization.py -v
```

Queries can be found from ``query_collections`` folder. It is easiest to install Memgraph and import the collections to Memgraph Lab. Then one can demonstrate the usage of the platform with the instances that are in the ``test`` folder.

This package also translates ZX-diagrams into Quimb tensor networks. It seems like this way, one can simulate a lot larger diagrams than with the tensor networks in PyZX. For this, see ``demo_quimb.ipynb``.

## Styles for MemGraph Lab

Apply the following style definitions in MemGraph Leb to match the style with PyZX:

```
@NodeStyle Equals(Property(node, "t"), 0) {
  color: #000000
  color-hover: #000000
  color-selected: #000000
}

@NodeStyle Equals(Property(node, "t"), 1) {
  color: #d6fccf
  color-hover: #d6fccf
  color-selected: #d6fccf
}

@NodeStyle Equals(Property(node, "t"), 2) {
  color: #f0908a
  color-hover: #f0908a
  color-selected: #f0908a
}

@NodeStyle Equals(Property(node, "t"), 3) {
  color: #ffd700
  color-hover: #ffd700
  color-selected: #ffd700
}
```

## Notes on differences compared to PyZX:

For reproducible scalability studies, realistic `full_reduce` workloads, and
PGFPlots/TikZ figures, see [the research evaluation suite](evaluation/RESEARCH.md).
It provides fixed study plans, repeated paired timings, applicability checks and
explicit correctness evidence instead of running the manual benchmark tests.

* There is possibility to perform spider fusion when red and green spider are connected with a Hadamard edge. As far as I know, ``spider_simp`` in PyZX does not detect these cases. Evaluating these cases is not implemented but it is noted.
* ``Bialg_simp`` is not clearly explained in the PyZX. I understand that it is the ''inverse'' of the bialgebra rule applied to all non-interacting instances.
* ``gadget_simp`` might have some side effects that depend on the phases?
