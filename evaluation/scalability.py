"""
Scientific scalability study: for each rewrite rule (and full_reduce), find the
"sweet spot" graph size where the database engine outperforms pyzx in
wall-clock RULE-EXECUTION time.

Method
------
We measure rule-execution time only (the graph is pre-loaded into the DB and
pre-built in pyzx; one-time load/build is excluded), with no correctness check
(speed only). For each rule we use a graph family G(N) chosen so the rule does
work that grows with N while the number of DB round-trips stays ~constant, so
the comparison reflects engine speed (Memgraph C++ query vs pyzx Python) rather
than round-trip overhead:

  * Batched rules (spider fusion, identity, hadamard) — a maximal set of
    disjoint trigger patterns is rewritten in ONE query, so we scale by the
    NUMBER of disjoint patterns N.
  * Single-application rules (lcomp, pivot, pivot_gadget, pivot_boundary,
    gadget fusion, supplementarity, copy, bialgebra) — one rewrite whose cost
    grows with the degree N of the matched spider(s) (O(N) or O(N^2) edges /
    spiders created in one query), so we scale that degree N.

We sweep N geometrically and report, per N: vertices, db time, pyzx time, ratio.
The crossover N* (first N with db < pyzx) is the sweet spot.

Usage
-----
    python -m evaluation.scalability                 # all rules + full_reduce
    python -m evaluation.scalability spider_fusion   # selected rules
    python -m evaluation.scalability full_reduce
"""
import io
import random
import sys
import time
from contextlib import redirect_stdout
from fractions import Fraction

import pyzx as zx

from zxdb.zxdb import ZXdb
from utils import zx_graph_to_db, pyzx_fixpoint
from pyzx.simplify import copy_simp as pyzx_copy_simp

GRAPH_ID = "example_graph"
B, Z, X = zx.VertexType.BOUNDARY, zx.VertexType.Z, zx.VertexType.X
S, H = zx.EdgeType.SIMPLE, zx.EdgeType.HADAMARD


# --------------------------------------------------------------- generators

def g_spider_fusion(n):
    """n disjoint fusable Z-Z simple-edge pairs (batched by the DB query)."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z1 = g.add_vertex(Z, phase=Fraction(1, 4))
        z2 = g.add_vertex(Z, phase=Fraction(1, 4)); bo = g.add_vertex(B)
        g.add_edge((bi, z1), S); g.add_edge((z1, z2), S); g.add_edge((z2, bo), S)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_identity(n):
    """n disjoint degree-2 phase-0 identity spiders."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z = g.add_vertex(Z, phase=Fraction(0)); bo = g.add_vertex(B)
        g.add_edge((bi, z), S); g.add_edge((z, bo), S)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_hadamard(n):
    """n disjoint H - Z(0) - H chains (even Hadamard path -> cancels)."""
    g = zx.Graph(); ins = []; outs = []
    for _ in range(n):
        bi = g.add_vertex(B); z = g.add_vertex(Z, phase=Fraction(0)); bo = g.add_vertex(B)
        g.add_edge((bi, z), H); g.add_edge((z, bo), H)
        ins.append(bi); outs.append(bo)
    g.set_inputs(tuple(ins)); g.set_outputs(tuple(outs)); return g


def g_lcomp(n):
    """One interior Z(pi/2) with n Z(pi/4) leaves on Hadamard edges. One local
    complementation removes the centre and creates K_n (~n^2/2 H-edges)."""
    g = zx.Graph()
    c = g.add_vertex(Z, phase=Fraction(1, 2))
    for _ in range(n):
        leaf = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((c, leaf), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot(n):
    """Two interior Pauli Z-spiders on a Hadamard edge, each with n exclusive
    Z(pi/4) leaves (H-edges). One pivot toggles the n x n bipartite edges."""
    g = zx.Graph()
    a = g.add_vertex(Z, phase=Fraction(0)); b = g.add_vertex(Z, phase=Fraction(0))
    g.add_edge((a, b), H)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((b, lb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot_gadget(n):
    """Interior Pauli Z(pi) - Hadamard - interior non-Pauli Z(pi/4), each with
    n Z(pi/4) leaves. One pivot-gadget application."""
    g = zx.Graph()
    a = g.add_vertex(Z, phase=Fraction(1)); b = g.add_vertex(Z, phase=Fraction(1, 4))
    g.add_edge((a, b), H)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((b, lb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_pivot_boundary(n):
    """Interior Pauli Z(pi) - Hadamard - Z spider that touches one boundary,
    each with n Z(pi/4) leaves. One pivot-boundary application."""
    g = zx.Graph(); ins = []
    a = g.add_vertex(Z, phase=Fraction(1))
    w = g.add_vertex(Z, phase=Fraction(1, 4))
    g.add_edge((a, w), H)
    bnd = g.add_vertex(B); g.add_edge((w, bnd), S); ins.append(bnd)
    for _ in range(n):
        la = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((a, la), H)
        lw = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((w, lw), H)
    g.set_inputs(tuple(ins)); g.set_outputs(()); return g


def g_gadget_fusion(n):
    """n phase gadgets sharing the SAME two targets -> all fuse into one."""
    g = zx.Graph()
    t1 = g.add_vertex(Z, phase=Fraction(0)); t2 = g.add_vertex(Z, phase=Fraction(0))
    for _ in range(n):
        axel = g.add_vertex(Z, phase=Fraction(0))
        leaf = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((axel, leaf), H)
        g.add_edge((axel, t1), H); g.add_edge((axel, t2), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_supplementarity(n):
    """Two non-Clifford Z-spiders (pi/4, 3pi/4) sharing n common neighbours;
    one supplementarity removes both and adds pi to the n neighbours."""
    g = zx.Graph()
    v = g.add_vertex(Z, phase=Fraction(1, 4)); w = g.add_vertex(Z, phase=Fraction(3, 4))
    for _ in range(n):
        u = g.add_vertex(Z, phase=Fraction(1, 4))
        g.add_edge((v, u), H); g.add_edge((w, u), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_copy(n):
    """Z(pi/4) hub with one X(pi) copy-leaf (simple edge) and n other Z leaves;
    one copy propagates the leaf through the hub to all n neighbours."""
    g = zx.Graph()
    hub = g.add_vertex(Z, phase=Fraction(1, 4))
    leaf = g.add_vertex(X, phase=Fraction(1)); g.add_edge((hub, leaf), S)
    for _ in range(n):
        nb = g.add_vertex(Z, phase=Fraction(1, 4)); g.add_edge((hub, nb), H)
    g.set_inputs(()); g.set_outputs(()); return g


def g_bialgebra(n):
    """Phase-0 Z-centre - simple edge - phase-0 X-centre; the Z-centre has n
    phase-0 X-neighbours and the X-centre n phase-0 Z-neighbours. One bialgebra
    application creates the n x n bipartite edges."""
    g = zx.Graph()
    z0 = g.add_vertex(Z, phase=Fraction(0)); x0 = g.add_vertex(X, phase=Fraction(0))
    g.add_edge((z0, x0), S)
    for _ in range(n):
        xn = g.add_vertex(X, phase=Fraction(0)); g.add_edge((z0, xn), S)
        zn = g.add_vertex(Z, phase=Fraction(0)); g.add_edge((x0, zn), S)
    g.set_inputs(()); g.set_outputs(()); return g


def g_full_reduce(n):
    """Random CNOT+H+T circuit on n qubits, depth ~ 20*n."""
    random.seed(1000 + n)
    return zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=n, depth=20 * n, p_had=0.2, p_t=0.25).to_graph()


# rule -> (generator, db_method_name, pyzx_callable, scale_label, N_values)
_P = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]   # pattern counts
_D = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]                       # degrees
_Q = [3, 4, 5, 6, 8, 10, 12, 14, 16]                                  # qubits

RULES = {
    "spider_fusion":   (g_spider_fusion,   "spider_fusion",
                        lambda g: pyzx_fixpoint(zx.spider_simp)(g), "patterns", _P),
    "identity":        (g_identity,        "remove_identities",
                        lambda g: pyzx_fixpoint(zx.id_simp)(g), "patterns", _P),
    "hadamard":        (g_hadamard,        "hadamard_cancel",
                        lambda g: pyzx_fixpoint(zx.id_simp)(g), "patterns", _P),
    "lcomp":           (g_lcomp,           "local_complementation_rule",
                        lambda g: pyzx_fixpoint(zx.lcomp_simp)(g), "degree", _D),
    "pivot":           (g_pivot,           "pivot_rule",
                        lambda g: pyzx_fixpoint(zx.pivot_simp)(g), "degree", _D),
    "pivot_gadget":    (g_pivot_gadget,    "pivot_gadget_rule",
                        lambda g: pyzx_fixpoint(zx.pivot_gadget_simp)(g), "degree", _D),
    "pivot_boundary":  (g_pivot_boundary,  "pivot_boundary_rule",
                        lambda g: pyzx_fixpoint(zx.pivot_boundary_simp)(g), "degree", _D),
    "gadget_fusion":   (g_gadget_fusion,   "phase_gadget_fusion_rule",
                        lambda g: pyzx_fixpoint(zx.gadget_simp)(g), "gadgets", _P),
    "supplementarity": (g_supplementarity, "supplementarity_simp",
                        lambda g: pyzx_fixpoint(zx.supplementarity_simp)(g), "neighbours", _D),
    "copy":            (g_copy,            "copy_simp",
                        lambda g: pyzx_fixpoint(pyzx_copy_simp)(g), "neighbours", _D),
    "bialgebra":       (g_bialgebra,       "bialgebra_simp",
                        lambda g: pyzx_fixpoint(zx.bialg_simp)(g), "degree", _D),
    "full_reduce":     (g_full_reduce,     "full_reduce",
                        lambda g: pyzx_fixpoint(lambda h: zx.full_reduce(h, quiet=True))(g),
                        "qubits", _Q),
}

# stop a sweep once either side exceeds this many seconds
TIME_CAP = 4.0


def _load(zxdb, g):
    with redirect_stdout(io.StringIO()):
        zx_graph_to_db(zxdb, g.copy(), graph_id=GRAPH_ID)


def sweep(zxdb, name):
    gen, method, pyzx_rule, label, ns = RULES[name]
    db_rule = getattr(zxdb, method)

    # warm query-plan cache and the pyzx code path on a tiny instance
    _load(zxdb, gen(2))
    db_rule(graph_id=GRAPH_ID)
    pyzx_rule(gen(2))

    print(f"\n### {name}  (scale = {label})")
    print(f"{label:>9} {'verts':>8} | {'db':>9} {'pyzx':>9} | ratio  winner")
    crossover = None
    for nval in ns:
        g = gen(nval)
        nverts = g.num_vertices()

        gp = g.copy()
        t = time.perf_counter(); pyzx_rule(gp); t_pyzx = time.perf_counter() - t

        zxdb.wipe_database()
        _load(zxdb, g)
        t = time.perf_counter(); db_rule(graph_id=GRAPH_ID); t_db = time.perf_counter() - t

        ratio = t_db / max(t_pyzx, 1e-9)
        winner = "DB" if t_db < t_pyzx else "pyzx"
        if t_db < t_pyzx and crossover is None:
            crossover = (nval, nverts)
        print(f"{nval:>9} {nverts:>8} | {t_db:>8.3f}s {t_pyzx:>8.4f}s | "
              f"{ratio:>5.1f}x {winner}", flush=True)
        if max(t_db, t_pyzx) > TIME_CAP:
            break

    if crossover:
        print(f"  --> crossover: DB wins from {label}={crossover[0]} "
              f"({crossover[1]} vertices)")
    else:
        print(f"  --> no crossover in range (pyzx faster throughout)")
    return crossover


def main(names):
    zxdb = ZXdb()
    results = {}
    try:
        for name in names:
            try:
                results[name] = sweep(zxdb, name)
            except Exception as e:
                print(f"  ERROR in {name}: {str(e)[:80]}")
        print("\n" + "=" * 60)
        print("CROSSOVER SUMMARY (DB starts beating pyzx at):")
        for name in names:
            c = results.get(name)
            print(f"  {name:<16} "
                  + (f"{RULES[name][3]}={c[0]}  ({c[1]} vertices)" if c
                     else "no crossover in tested range"))
    finally:
        zxdb.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = args or list(RULES.keys())
    unknown = [n for n in names if n not in RULES]
    if unknown:
        print(f"Unknown: {unknown}. Available: {list(RULES.keys())}")
        sys.exit(2)
    main(names)
