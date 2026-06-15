# zxdb query modules — in-process pivot_gadget

`full_reduce_with_query_modules` (in `zxdb/zxdb.py`) is an isolated variant of
`full_reduce` that runs the **pivot_gadget** rule inside Memgraph as a custom
query module instead of via per-application Cypher. Everything else stays Cypher.

## Why

pivot_gadget is the one rule where the Cypher implementation lags pyzx: pyzx's
matcher finds a **maximal set of vertex-disjoint gadget matches in one O(E)
pass** (greedy: mark each match's neighbourhood consumed, skip overlapping
candidates) and applies them all, whereas the Cypher rule applies one per query
(`LIMIT 1`) and re-scans every time. That greedy is *imperative in-process* — not
expressible cheaply in declarative Cypher (mutual-min under-selects) nor via MAGE
coloring (the conflict graph is quadratic on dense inputs). A custom query module
runs in-process, so it can do exactly pyzx's greedy, with the heavy O(deg^2)
edge toggles done natively.

## Files

- `zxqmcpp.cpp` — **C++** module (`zxqmcpp.pivot_gadget_fixpoint()`), the
  performant one. Ports pyzx `match_pivot_gadget` + `pivot`; runs the whole
  pivot_gadget fixpoint in one CALL.
- `zxqm.py` — **Python** mgp port (`zxqm.pivot_gadget_fixpoint(graph_id)`), a
  reference. Correct but slower (mgp per-op overhead exceeds the round-trip
  savings); kept for comparison.

## Result (Memgraph 3.6.2, this machine)

Correct: 112/112 full_reduce eval + 8/8 small tensor-vs-pyzx. On dense
gadget-heavy circuits (q28, span None) full_reduce_with_query_modules is
**~2–2.5x faster than the Cypher full_reduce and ~1.4–2.2x faster than pyzx**,
including degree 12 / twInt~22 where the Cypher full_reduce previously *tied/lost*
to pyzx — the C++ module wins there too (0.48x). So it removes the one regime
where the DB lagged pyzx.

## Deploy (one-time per container)

The module must live in Memgraph's `--query-modules-directory`
(`/usr/lib/memgraph/query_modules`). The `memgraph-mage` image ships the headers
(`/usr/include/memgraph/mgp.hpp`) but no compiler, so install g++ once.

```sh
# from repo root, with the 'memgraph' container running:
docker exec -u root memgraph apt-get update -qq
docker exec -u root memgraph apt-get install -y -qq g++
docker cp zxdb_qm/zxqmcpp.cpp memgraph:/tmp/zxqmcpp.cpp
docker exec -u root memgraph g++ -std=c++20 -fPIC -shared \
    -I/usr/include/memgraph /tmp/zxqmcpp.cpp \
    -o /usr/lib/memgraph/query_modules/zxqmcpp.so
docker cp zxdb_qm/zxqm.py memgraph:/usr/lib/memgraph/query_modules/zxqm.py   # optional (Python port)
docker exec -u root memgraph chmod 644 /usr/lib/memgraph/query_modules/zxqm.py
```

Then `ZXdb.ensure_query_module()` runs `CALL mg.load_all()` and verifies
`zxqmcpp.pivot_gadget_fixpoint` is available. The `.so` persists in the container
across restarts (re-run the compile only if the container is recreated).
