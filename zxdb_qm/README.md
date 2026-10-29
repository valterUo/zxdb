# Native Memgraph module

The hybrid engine uses `zxqmcpp` for pivot, local complementation, gadget pivot,
boundary pivot and supplementarity; other rewrites remain in Python/Cypher.
Use a dedicated Memgraph 3.14.0 container. The commands below assume its name is
`zxdb-reproduce` (as in the root README).

## Build

The original build used g++ 13.3.0 and the flags below. Exact image, source,
module and missing-header hashes are recorded in
[build.json](../results/memgraph-3.14/build.json).

```sh
docker exec -u root zxdb-reproduce apt-get update
docker exec -u root zxdb-reproduce apt-get install -y g++ curl
docker exec -u root zxdb-reproduce curl -fsSL https://raw.githubusercontent.com/memgraph/memgraph/v3.14.0/include/text_search_config.hpp -o /usr/include/memgraph/text_search_config.hpp
docker cp zxdb_qm/zxqmcpp.cpp zxdb-reproduce:/tmp/zxqmcpp.cpp
docker exec -u root zxdb-reproduce g++ -O3 -std=c++20 -fPIC -shared -I/usr/include/memgraph /tmp/zxqmcpp.cpp -o /tmp/zxqmcpp-new.so
docker exec -u root zxdb-reproduce mv /tmp/zxqmcpp-new.so /usr/lib/memgraph/query_modules/zxqmcpp.so
docker restart zxdb-reproduce
```

The header download supplies a header missing from that image's SDK. A restart
loads the new binary; `mg.load_all()` alone may leave the old library cached.
The evaluation correctness gate checks all six required procedures before timing.
The original measured binary is retained at `results/zxqmcpp-rerun-20261029.so`.

Follow the [root README](../README.md) to run correctness and benchmarks.
Procedures require exclusive access to their graph and reject cross-graph edges,
self-loops and parallel edges; the wrapper normalizes parallel edges first.

`zxqm.py` is an optional Python procedure retained with its regression tests.
It is not used by the published three-engine comparison. To check it, deploy it
to the container's query-module directory, restart, and run the test suite with
`ZXDB_LIVE=1` and `ZXDB_PYTHON_MODULE=1` against the dedicated database.
