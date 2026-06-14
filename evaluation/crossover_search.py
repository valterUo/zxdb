"""
Crossover search: which CIRCUIT FAMILIES make the DB full_reduce beat pyzx?

The per-rule study (evaluation/SCALABILITY.md) showed the DB wins when a rule
does ONE LARGE rewrite (high-degree spider, O(N) / O(N^2) work in a single
Cypher query) and loses when it does MANY SMALL rewrites (each ~1 round-trip).
A random deep circuit is the worst case: it fragments into hundreds of tiny
pivots, so the DB carries a ~1.5-2x round-trip constant and never crosses 1x.

This script tests the opposite end: circuit families whose reduction is
dominated by FEW, LARGE rewrites. Hypothesis: the DB crosses over on WIDE
CLIFFORD-dominated circuits (spider degree ~ O(qubits), graph collapses in
relatively few high-degree pivots/lcomps) and on structured high-connectivity
circuits (QAOA), while losing on DEEP-NARROW and T-HEAVY circuits that stay
sparse and fragment.

For each family we sweep the relevant dimension and report, per size:
  input verts | DB full_reduce time | round-trips | pyzx time | ratio | winner
plus the final reduced vertex count. Correctness of the DB reduction is
verified (tensor/sampled vs pyzx) on the SMALLEST instance of every family, so
a "win" is a win on a correct reduction.

Usage:
    python -m evaluation.crossover_search                 # all families
    python -m evaluation.crossover_search clifford_wide   # selected
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
from evaluation.harness import verify_pair
from evaluation.benchmark_perf import _count_round_trips

GRAPH_ID = "example_graph"
PYZX_FR = pyzx_fixpoint(lambda h: zx.full_reduce(h, quiet=True))

# stop a family sweep once either side exceeds this many seconds
TIME_CAP = 9.0


# ----------------------------------------------------------------- families
# Each generator takes the sweep value n and returns a pyzx graph with
# inputs/outputs set. Seeds are derived from n for reproducibility.

def f_clifford_wide(n):
    """WIDE genuine-Clifford circuit: qubits = n, depth = 8*n. Predicted DB
    winner: reduces to a Clifford normal form via high-degree pivots/lcomps."""
    return zx.generate.cliffords(n, 8 * n, seed=100 + n)


def f_clifford_deep(n):
    """DEEP-NARROW Clifford: 6 qubits, depth = n. Predicted DB loser: stays
    narrow (low degree) so reduction is many small rewrites."""
    return zx.generate.cliffords(6, n, seed=200 + n)


def f_near_clifford(n):
    """Wide near-Clifford: qubits = n, depth = 8*n, 10% T. A few gadgets on top
    of a Clifford backbone."""
    return zx.generate.cliffordT(n, 8 * n, p_t=0.1, seed=300 + n)


def f_t_heavy(n):
    """Wide T-heavy: qubits = n, depth = 8*n, 50% T. Predicted DB loser: many
    gadgets -> many small pivot_gadget/gadget_fusion rewrites."""
    return zx.generate.cliffordT(n, 8 * n, p_t=0.5, seed=400 + n)


def _qaoa(qubits, p_layers, degree, gamma, seed):
    """QAOA-style ansatz: p layers of (ZZ cost on a random d-regular graph,
    then RX mixer). gamma non-Clifford -> phase-gadget structure."""
    rng = random.Random(seed)
    # build a random degree-`degree` graph by random matching rounds
    edges = set()
    for _ in range(degree):
        perm = list(range(qubits)); rng.shuffle(perm)
        for i in range(0, qubits - 1, 2):
            a, b = perm[i], perm[i + 1]
            if a != b:
                edges.add((min(a, b), max(a, b)))
    c = zx.Circuit(qubits)
    for q in range(qubits):
        c.add_gate("HAD", q)
    for _ in range(p_layers):
        for (a, b) in edges:
            c.add_gate("CNOT", a, b)
            c.add_gate("ZPhase", b, phase=gamma)
            c.add_gate("CNOT", a, b)
        for q in range(qubits):
            c.add_gate("XPhase", q, phase=Fraction(1, 2))
    return c.to_graph()


def f_qaoa(n):
    """QAOA on n qubits, 3 layers, 3-regular graph, gamma = pi/4."""
    return _qaoa(n, p_layers=3, degree=3, gamma=Fraction(1, 4), seed=500 + n)


def f_qaoa_dense(n):
    """Denser QAOA: n qubits, 2 layers, ~n/2-regular graph (high connectivity
    -> high-degree spiders), gamma = pi/4."""
    return _qaoa(n, p_layers=2, degree=max(2, n // 2), gamma=Fraction(1, 4),
                 seed=600 + n)


# family -> (generator, sweep values)
_WIDTH = [8, 16, 24, 32, 40, 48, 56, 64]
_DEPTH = [50, 100, 200, 400, 800, 1600, 3200]

FAMILIES = {
    "clifford_wide":  (f_clifford_wide,  _WIDTH),
    "clifford_deep":  (f_clifford_deep,  _DEPTH),
    "near_clifford":  (f_near_clifford,  _WIDTH),
    "t_heavy":        (f_t_heavy,        _WIDTH),
    "qaoa":           (f_qaoa,           _WIDTH),
    "qaoa_dense":     (f_qaoa_dense,     [8, 12, 16, 20, 24, 28, 32]),
}


def _load(zxdb, g):
    with redirect_stdout(io.StringIO()):
        zx_graph_to_db(zxdb, g.copy(), graph_id=GRAPH_ID)


def sweep(zxdb, rt, name, verify_smallest=True):
    gen, ns = FAMILIES[name]
    print(f"\n### {name}")
    print(f"{'n':>5} {'in_v':>6} {'out_v':>6} | {'db':>8} {'rt':>4} | "
          f"{'pyzx':>9} | ratio  winner  {'check' if verify_smallest else ''}")
    crossover = None
    first = True
    for n in ns:
        try:
            g = gen(n)
        except Exception as e:
            print(f"{n:>5}  gen FAIL: {str(e)[:50]}"); continue
        nin = g.num_vertices()
        original = g.copy()

        # pyzx side
        pg = g.copy()
        t = time.perf_counter(); PYZX_FR(pg); t_pyzx = time.perf_counter() - t
        nout = pg.num_vertices()

        # DB side (warm this size once, then time)
        try:
            zxdb.wipe_database()
            _load(zxdb, g)
            zxdb.full_reduce(GRAPH_ID)
            zxdb.wipe_database()
            _load(zxdb, g)
            rt["n"] = 0
            t = time.perf_counter(); zxdb.full_reduce(GRAPH_ID)
            t_db = time.perf_counter() - t
            nrt = rt["n"]
        except Exception as e:
            print(f"{n:>5} {nin:>6} {'':>6} | DB FAIL: {str(e)[:40]}")
            break

        ratio = t_db / max(t_pyzx, 1e-9)
        winner = "DB  " if t_db < t_pyzx else "pyzx"
        if t_db < t_pyzx and crossover is None:
            crossover = (n, nin)

        chk = ""
        if verify_smallest and first:
            try:
                rt_was = rt["n"]
                db_g = zxdb.export_graphdb_to_zx_graph(GRAPH_ID, "example.json")
                ok, level = verify_pair(db_g, pg, original=original)
                rt["n"] = rt_was
                chk = f"{'OK' if ok else 'FAIL'}[{level}]"
            except Exception as e:
                chk = f"chk-err:{str(e)[:20]}"
        first = False

        print(f"{n:>5} {nin:>6} {nout:>6} | {t_db:>7.3f}s {nrt:>4} | "
              f"{t_pyzx:>8.4f}s | {ratio:>5.2f}x {winner}  {chk}", flush=True)
        if max(t_db, t_pyzx) > TIME_CAP:
            print(f"      (time cap {TIME_CAP}s reached)")
            break

    if crossover:
        print(f"  --> DB WINS from n={crossover[0]} ({crossover[1]} input verts)")
    else:
        print(f"  --> no crossover in tested range")
    return crossover


def main(names):
    zxdb = ZXdb()
    rt = _count_round_trips()
    results = {}
    try:
        for name in names:
            try:
                results[name] = sweep(zxdb, rt, name)
            except Exception as e:
                print(f"  ERROR in {name}: {str(e)[:80]}")
        print("\n" + "=" * 64)
        print("CROSSOVER SUMMARY (DB full_reduce starts beating pyzx at):")
        for name in names:
            c = results.get(name)
            print(f"  {name:<16} "
                  + (f"n={c[0]}  ({c[1]} input verts)" if c
                     else "no crossover in tested range"))
    finally:
        rt["restore"]()
        zxdb.wipe_database()
        zxdb.close()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    names = args or list(FAMILIES.keys())
    unknown = [n for n in names if n not in FAMILIES]
    if unknown:
        print(f"Unknown: {unknown}. Available: {list(FAMILIES.keys())}")
        sys.exit(2)
    main(names)
