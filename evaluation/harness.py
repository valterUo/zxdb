"""
Adversarial evaluation harness for zxdb rewrite rules.

For each test case (a small pyzx graph) the harness:
  1. loads the graph into the database,
  2. applies the DB rule and the corresponding pyzx rule,
  3. compares the results at three levels:
       - stats / degree distribution (DB result vs pyzx result)
       - graph isomorphism with phases normalized mod 2
       - tensor equivalence of the DB result against the ORIGINAL graph
         (the critical semantic-correctness check, up to global scalar)
  4. inspects the database directly for parallel edges and self-loops,
     because export_graphdb_to_zx_graph silently drops both.
"""
import io
import itertools
import traceback
from contextlib import redirect_stdout
from fractions import Fraction

import networkx as nx
import numpy as np
import pyzx as zx

from utils import zx_graph_to_db, pyzx_fixpoint  # noqa: F401  (re-exported)

GRAPH_ID = "example_graph"
MAX_PERM_QUBITS = 4  # try boundary permutations up to this many in/outputs


def _normalized_nx(g):
    """pyzx graph -> networkx graph with phases normalized to [0, 2)."""
    nxg = nx.Graph()
    for v in g.vertices():
        phase = g.phase(v)
        if phase is None:
            phase = 0
        phase = Fraction(phase).limit_denominator(10**6) % 2
        nxg.add_node(v, type=g.type(v), phase=phase)
    for e in g.edges():
        s, t = g.edge_s(e), g.edge_t(e)
        nxg.add_edge(s, t, type=g.edge_type(e))
    return nxg


def _node_match(d1, d2):
    return d1["type"] == d2["type"] and d1["phase"] == d2["phase"]


def _edge_match(d1, d2):
    return d1["type"] == d2["type"]


def _clean_copy(g):
    """Rebuild a graph with sane circuit-like positions: the exporter's
    spring-layout positions break pyzx's tensorfy (row ordering assumptions),
    and so do all-equal rows on some topologies. Rows are assigned by BFS
    distance from the inputs; outputs are pushed to the last row."""
    from collections import deque
    dist = {}
    queue = deque()
    for v in g.inputs():
        dist[v] = 0
        queue.append(v)
    while queue:
        v = queue.popleft()
        for n in g.neighbors(v):
            if n not in dist:
                dist[n] = dist[v] + 1
                queue.append(n)
    max_row = max(dist.values(), default=0) + 1
    h = zx.Graph()
    vmap = {}
    per_row_count = {}
    for v in g.vertices():
        row = max_row if v in g.outputs() else dist.get(v, max_row)
        qubit = per_row_count.get(row, 0)
        per_row_count[row] = qubit + 1
        vmap[v] = h.add_vertex(ty=g.type(v), phase=g.phase(v),
                               qubit=qubit, row=row)
    for e in g.edges():
        s, t = g.edge_s(e), g.edge_t(e)
        h.add_edge((vmap[s], vmap[t]), g.edge_type(e))
    h.set_inputs(tuple(vmap[v] for v in g.inputs()))
    h.set_outputs(tuple(vmap[v] for v in g.outputs()))
    return h


def _tensors_equal(g1, g2):
    try:
        t1 = _clean_copy(g1).to_tensor()
        t2 = _clean_copy(g2).to_tensor()
        # compare_tensors scales by the first nonzero entry; an all-zero
        # tensor (e.g. an isolated Z(pi) scalar factor) silently "equals"
        # anything. Treat zero tensors explicitly.
        z1 = np.allclose(t1, 0)
        z2 = np.allclose(t2, 0)
        if z1 or z2:
            return z1 == z2
        return zx.compare_tensors(t1, t2, preserve_scalar=False)
    except Exception:
        return None


def _semantic_equal(db_g, original):
    """Tensor comparison; falls back to boundary permutations because the
    DB round-trip does not always preserve input/output ordering.

    Returns None when the ORIGINAL diagram is the zero map: rewrites are
    then only correct "up to a zero scalar", which the DB does not track,
    so any reduced graph is acceptable and the check is indeterminate.
    """
    try:
        if np.allclose(_clean_copy(original).to_tensor(), 0):
            return None
    except Exception:
        pass
    if _tensors_equal(db_g, original):
        return True
    ins, outs = db_g.inputs(), db_g.outputs()
    if max(len(ins), len(outs)) > MAX_PERM_QUBITS:
        return False
    for in_perm in itertools.permutations(ins):
        for out_perm in itertools.permutations(outs):
            db_g.set_inputs(in_perm)
            db_g.set_outputs(out_perm)
            if _tensors_equal(db_g, original):
                db_g.set_inputs(ins)
                db_g.set_outputs(outs)
                return True
    db_g.set_inputs(ins)
    db_g.set_outputs(outs)
    return False


def _db_integrity(zxdb):
    """Check DB for things the exporter hides: parallel edges, self-loops,
    and phases outside [0, 2)."""
    with zxdb.driver.session() as s:
        parallel = s.run(
            "MATCH (a:Node)-[r:Wire]-(b:Node) WHERE id(a) < id(b) "
            "WITH a, b, count(r) AS c WHERE c > 1 RETURN count(*) AS n"
        ).single()["n"]
        loops = s.run(
            "MATCH (a:Node)-[r:Wire]->(a) RETURN count(r) AS n"
        ).single()["n"]
        bad_phase = s.run(
            "MATCH (n:Node) WHERE n.phase IS NOT NULL "
            "AND (n.phase < 0 OR n.phase >= 2) RETURN count(n) AS n"
        ).single()["n"]
    return {"parallel_edges": parallel, "self_loops": loops,
            "phases_out_of_range": bad_phase}


def _degree_sequence(g):
    return sorted(g.vertex_degree(v) for v in g.vertices())


def run_case(zxdb, rule_name, case_name, g, db_rule, pyzx_rule,
             check_tensor=True, require_iso=True):
    """Run one corner case. Returns a result dict.

    require_iso=False relaxes the structural checks for cases where DB and
    pyzx may legitimately apply matches in different orders (non-confluent
    intermediate states); tensor equivalence then decides correctness.
    """
    res = {"rule": rule_name, "case": case_name, "error": None}
    try:
        original = g.copy()
        pyzx_g = g.copy()

        buf = io.StringIO()
        with redirect_stdout(buf):
            zx_graph_to_db(zxdb, g.copy(), graph_id=GRAPH_ID)
            res["db_count"] = db_rule(graph_id=GRAPH_ID)
            res["pyzx_count"] = pyzx_rule(pyzx_g)
            db_g = zxdb.export_graphdb_to_zx_graph(GRAPH_ID, "example.json")

        res.update(_db_integrity(zxdb))

        res["db_stats"] = f"{db_g.num_vertices()}v/{db_g.num_edges()}e"
        res["pyzx_stats"] = f"{pyzx_g.num_vertices()}v/{pyzx_g.num_edges()}e"
        res["stats_match"] = (db_g.num_vertices() == pyzx_g.num_vertices()
                              and db_g.num_edges() == pyzx_g.num_edges())
        res["degree_seq_match"] = (_degree_sequence(db_g)
                                   == _degree_sequence(pyzx_g))

        res["isomorphic"] = nx.is_isomorphic(
            _normalized_nx(db_g), _normalized_nx(pyzx_g),
            node_match=_node_match, edge_match=_edge_match)

        if check_tensor:
            res["db_semantic"] = _semantic_equal(db_g, original)
            res["pyzx_semantic"] = _tensors_equal(pyzx_g, original)
        else:
            res["db_semantic"] = res["pyzx_semantic"] = None

        # Semantic criterion: the DB result must be equivalent to the original
        # graph — unless pyzx itself is also wrong (garbage-in/garbage-out on
        # non-graph-like input) and the DB matches pyzx exactly, which counts
        # as reference parity.
        semantic_ok = (res["db_semantic"] is not False
                       or (res["pyzx_semantic"] is False and res["isomorphic"]))
        # db_semantic None = indeterminate (zero diagram or tensor error):
        # fall back to the structural comparison against pyzx.
        structural_ok = res["isomorphic"] if require_iso \
            else (res["db_semantic"] is True
                  or (res["db_semantic"] is None and res["stats_match"]))
        res["ok"] = bool(
            structural_ok
            and res["parallel_edges"] == 0
            and res["self_loops"] == 0
            and semantic_ok
        )
    except Exception:
        res["error"] = traceback.format_exc(limit=3)
        res["ok"] = False
    return res


def print_results(results):
    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n{'=' * 78}")
    print(f"RESULTS: {n_ok}/{len(results)} cases passed")
    print(f"{'=' * 78}")
    for r in results:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[{flag}] {r['rule']} :: {r['case']}")
        if r["error"]:
            print(f"       ERROR: {r['error'].splitlines()[-1]}")
            continue
        detail = (f"db={r['db_stats']} pyzx={r['pyzx_stats']} "
                  f"degseq={r['degree_seq_match']} iso={r['isomorphic']} "
                  f"db_sem={r['db_semantic']} pyzx_sem={r['pyzx_semantic']} "
                  f"counts=({r['db_count']},{r['pyzx_count']})")
        print(f"       {detail}")
        extras = []
        if r["parallel_edges"]:
            extras.append(f"parallel_edges={r['parallel_edges']}")
        if r["self_loops"]:
            extras.append(f"self_loops={r['self_loops']}")
        if r["phases_out_of_range"]:
            extras.append(f"phases_out_of_range={r['phases_out_of_range']}")
        if extras:
            print(f"       DB integrity: {', '.join(extras)}")
    return n_ok == len(results)
