# Native database rewrites

`ZXdb.full_reduce_native(graph_id)` uses C++ query procedures for interior pivot,
local complementation, both boundary-pivot variants, gadget pivot and supplementarity.
Other rules and the reduction schedule
remain in Python/Cypher. All graph mutations execute inside Memgraph; Python
imports and exports the graph outside the rewrite timer.

The scoped procedures are `pivot`, `lcomp`, `gadget`, `pauli_boundary`,
`boundary` and `supplementarity` in the `zxqmcpp` module. Each takes `graph_id`
and runs its rule to a fixed point. They reject cross-graph edges, self-loops and
parallel edges before mutation. The full-reduction wrapper normalizes parallel
edges. Use exclusive access to the selected graph.

## Build and activate

Build against the headers from the running Memgraph version. The commands below
assume a dedicated container named `memgraph` with g++ installed. Install g++ in
that container if needed. Compile to a temporary file before replacing the module.

The latest validation uses `memgraph/memgraph:3.14.0`. That image's SDK omitted
`text_search_config.hpp`, included by `mgp.hpp`. If compilation reports that
missing header, copy the matching [v3.14.0 header](https://raw.githubusercontent.com/memgraph/memgraph/v3.14.0/include/text_search_config.hpp)
to `/usr/include/memgraph/` before compiling. The exact validation header and
compiler/image hashes are recorded in `results/memgraph-3.14/build.json`.

```sh
docker cp zxdb_qm/zxqmcpp.cpp memgraph:/tmp/zxqmcpp.cpp
docker exec -u root memgraph g++ -O3 -std=c++20 -fPIC -shared -I/usr/include/memgraph /tmp/zxqmcpp.cpp -o /tmp/zxqmcpp-new.so
docker exec -u root memgraph mv /tmp/zxqmcpp-new.so /usr/lib/memgraph/query_modules/zxqmcpp.so
docker restart memgraph
```

A restart unloads cached libraries; `mg.load_all()` alone did not replace the old
library in the tested container. Check `CALL mg.procedures()` for all six scoped
procedures before running the evaluation.

## Evaluate

From the repository root, against an empty dedicated database:

```sh
python evaluate.py --variants native --full-reduce-only --qubits 32 64 128 --depth-factor 20 --seeds 3 --output results/native
```

This retains every correctness input and adds native regression tests. Timings,
checks and source hashes use the same report format as the Cypher comparison.
Large circuit tensors can remain inconclusive; raw timing alone is not evidence
of correctness. Database scalar tracking remains an explicit limitation.

`zxqm.py` and the zero-argument C++ gadget procedure are older comparison paths;
their Python wrapper requires that no other graph is present. Historical timing
claims are in the backup, not evidence for the current implementation.
