"""
Wall-time benchmark of the zxdb full_reduce pipeline against pyzx.

Generates random CNOT+H+T circuits of growing size, runs the DB pipeline and
pyzx full_reduce on identical inputs, and reports wall times and the DB/pyzx
ratio. Each size is warmed up once (cold runs pay query-plan compilation and
connection setup) and then timed.

Findings (see evaluation/README.md "Performance"):
  * The DB issues ~100 Cypher round-trips per full_reduce, almost independent
    of graph size, so it carries a fixed ~0.2-0.3s overhead. pyzx runs entirely
    in-process in microseconds per rewrite.
  * Because that overhead is fixed while pyzx's per-rewrite cost grows with the
    graph, the DB/pyzx ratio shrinks sharply with size (≈40x at ~70 vertices
    down to ≈6x at ~730 vertices on this machine).

Usage:
    python -m evaluation.benchmark_perf            # default sizes (fast)
    python -m evaluation.benchmark_perf --profile  # per-rule round-trip/time
    python -m evaluation.benchmark_perf --big       # include larger sizes
"""
import io
import random
import sys
import time
from contextlib import redirect_stdout

import pyzx as zx

from zxdb.zxdb import ZXdb
from utils import zx_graph_to_db, pyzx_fixpoint
from evaluation.harness import verify_pair

GRAPH_ID = "example_graph"

DEFAULT_SIZES = [(4, 40), (5, 70), (6, 100), (7, 140), (8, 200)]
BIG_SIZES = [(10, 300), (12, 450), (14, 650)]


def make_circuit(seed, qubits, depth):
    random.seed(seed)
    c = zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=qubits, depth=depth, p_had=0.2, p_t=0.25)
    return c.to_graph()


def load(zxdb, g):
    with redirect_stdout(io.StringIO()):
        zx_graph_to_db(zxdb, g.copy(), graph_id=GRAPH_ID)


def _count_round_trips():
    """Patch the neo4j Transaction.run to count round-trips. Returns a dict
    with a mutable 'n' and a restore() callable."""
    import neo4j._sync.work.transaction as txmod
    orig = txmod.TransactionBase.run
    state = {"n": 0}

    def patched(self, q, *a, **k):
        state["n"] += 1
        return orig(self, q, *a, **k)

    txmod.TransactionBase.run = patched
    state["restore"] = lambda: setattr(txmod.TransactionBase, "run", orig)
    return state


def run_benchmark(sizes, verify=False):
    zxdb = ZXdb()
    header = (f"{'size':>10} {'verts':>6} | {'db':>8} {'rt':>4} | "
             f"{'pyzx':>9} | ratio")
    if verify:
        header += " | correctness (db vs pyzx)"
    print(header)
    rt = _count_round_trips()
    try:
        for i, (qubits, depth) in enumerate(sizes):
            g = make_circuit(8000 + i, qubits, depth)
            nverts = g.num_vertices()
            original = g.copy()

            pyzx_g = g.copy()
            t = time.perf_counter()
            pyzx_fixpoint(lambda h: zx.full_reduce(h, quiet=True))(pyzx_g)
            t_pyzx = time.perf_counter() - t

            load(zxdb, g)
            zxdb.full_reduce(GRAPH_ID)          # warm up this size
            load(zxdb, g)
            rt["n"] = 0
            t = time.perf_counter()
            zxdb.full_reduce(GRAPH_ID)
            t_db = time.perf_counter() - t

            line = (f"q{qubits:<2} d{depth:<5} {nverts:>6} | "
                    f"{t_db:>7.3f}s {rt['n']:>4} | {t_pyzx:>8.4f}s | "
                    f"{t_db / max(t_pyzx, 1e-9):>4.1f}x")
            if verify:
                rt_was = rt["n"]
                db_g = zxdb.export_graphdb_to_zx_graph(GRAPH_ID, "example.json")
                ok, level = verify_pair(db_g, pyzx_g, original=original)
                rt["n"] = rt_was
                line += f" | {'OK ' if ok else 'FAIL'} [{level}]"
            print(line, flush=True)
    finally:
        rt["restore"]()
        zxdb.close()


def profile(qubits=6, depth=100):
    """Per-rule round-trip and time breakdown of one full_reduce."""
    zxdb = ZXdb()
    rt = _count_round_trips()
    try:
        g = make_circuit(8100, qubits, depth)
        load(zxdb, g)
        zxdb.full_reduce(GRAPH_ID)  # warm

        from collections import defaultdict
        rounds, times = defaultdict(int), defaultdict(float)

        def wrap(name, fn):
            rt["n"] = 0
            t = time.perf_counter()
            r = fn(GRAPH_ID)
            rounds[name] += rt["n"]
            times[name] += time.perf_counter() - t
            return r

        load(zxdb, g)
        # mirror full_reduce composition with timing hooks
        wrap("spider_fusion", zxdb.spider_fusion)
        wrap("to_gh", zxdb.to_gh)
        while True:
            n = (wrap("remove_identities", zxdb.remove_identities)
                 + wrap("spider_fusion", zxdb.spider_fusion)
                 + wrap("pivot_rule", zxdb.pivot_rule)
                 + wrap("lcomp", zxdb.local_complementation_rule))
            if n == 0:
                break
        wrap("pivot_gadget", zxdb.pivot_gadget_rule)
        for _ in range(100):
            while True:
                n = (wrap("remove_identities", zxdb.remove_identities)
                     + wrap("spider_fusion", zxdb.spider_fusion)
                     + wrap("pivot_rule", zxdb.pivot_rule)
                     + wrap("lcomp", zxdb.local_complementation_rule))
                if n == 0:
                    break
            wrap("pivot_boundary", zxdb.pivot_boundary_rule)
            pi = wrap("gadget_fusion", zxdb.phase_gadget_fusion_rule)
            while True:
                n = (wrap("remove_identities", zxdb.remove_identities)
                     + wrap("spider_fusion", zxdb.spider_fusion)
                     + wrap("pivot_rule", zxdb.pivot_rule)
                     + wrap("lcomp", zxdb.local_complementation_rule))
                if n == 0:
                    break
            pk = wrap("copy_simp", zxdb.copy_simp)
            pl = wrap("supplementarity", zxdb.supplementarity_simp)
            pj = wrap("pivot_gadget", zxdb.pivot_gadget_rule)
            if pi + pj + pk + pl == 0:
                wrap("remove_isolated", zxdb.remove_isolated_vertices)
                break

        print(f"per-rule profile (q{qubits} d{depth}, "
              f"{g.num_vertices()} input verts):")
        print(f"{'rule':<20} {'round-trips':>11} {'time':>9}")
        for name in sorted(rounds, key=lambda x: -times[x]):
            print(f"{name:<20} {rounds[name]:>11} {1000 * times[name]:>7.0f}ms")
        print(f"{'TOTAL':<20} {sum(rounds.values()):>11} "
              f"{1000 * sum(times.values()):>7.0f}ms")
    finally:
        rt["restore"]()
        zxdb.close()


if __name__ == "__main__":
    if "--profile" in sys.argv:
        profile()
    else:
        sizes = DEFAULT_SIZES + (BIG_SIZES if "--big" in sys.argv else [])
        run_benchmark(sizes, verify="--verify" in sys.argv)
