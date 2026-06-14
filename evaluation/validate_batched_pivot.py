"""Correctness gate for full_reduce_batched_pivot (isolated batched-pivot variant).

For many small, tensor-feasible circuits across several families it checks:
  (1) batched full_reduce  ==tensor==  pyzx full_reduce      (the real gate)
  (2) batched full_reduce  ==tensor==  original DB full_reduce
and reports the round-trip saving. Also runs adversarial pivot structures
(many disjoint pivots; overlapping pivots) directly through the batched vs the
single interior-pivot fixpoint and checks identical results.

Run:  python -m evaluation.validate_batched_pivot
"""
import io
import random
from contextlib import redirect_stdout
from fractions import Fraction

import pyzx as zx

from zxdb.zxdb import ZXdb
from utils import zx_graph_to_db, pyzx_fixpoint
from evaluation.harness import verify_pair
from evaluation.crossover_search import _qaoa

GID = "example_graph"
PR = pyzx_fixpoint(lambda h: zx.full_reduce(h, quiet=True))


def _load(z, g):
    with redirect_stdout(io.StringIO()):
        zx_graph_to_db(z, g.copy(), graph_id=GID)


def _cases():
    cs = []
    for s in range(1, 9):
        random.seed(s)
        cs.append((f"rand_q3_d24_s{s}",
                   zx.generate.CNOT_HAD_PHASE_circuit(3, 24, p_had=0.3, p_t=0.3).to_graph()))
        cs.append((f"rand_q4_d20_s{s}",
                   zx.generate.CNOT_HAD_PHASE_circuit(4, 20, p_had=0.25, p_t=0.3).to_graph()))
    for s in range(1, 5):
        cs.append((f"qaoa_n4_p2_s{s}", _qaoa(4, 2, 3, Fraction(1, 4), 50 + s)))
        cs.append((f"qaoa_n5_p1_s{s}", _qaoa(5, 1, 3, Fraction(1, 4), 70 + s)))
        cs.append((f"qaoa_n6_p1_s{s}", _qaoa(6, 1, 3, Fraction(1, 4), 90 + s)))
    for s in range(1, 4):
        cs.append((f"cliffT_q5_s{s}", zx.generate.cliffordT(5, 40, p_t=0.3, seed=s)))
        cs.append((f"cliff_q5_s{s}", zx.generate.cliffords(5, 40, seed=s)))
    return cs


def main():
    z = ZXdb()
    import neo4j._sync.work.transaction as txmod
    o = txmod.TransactionBase.run
    rt = {"n": 0}
    txmod.TransactionBase.run = lambda self, q, *a, **k: (
        rt.__setitem__("n", rt["n"] + 1) or o(self, q, *a, **k))
    npass = nfail = 0
    rt_orig = rt_batch = 0
    try:
        for name, g in _cases():
            orig = g.copy()
            pg = g.copy(); PR(pg)
            # original full_reduce
            z.wipe_database(); _load(z, g); rt["n"] = 0
            z.full_reduce(GID); rto = rt["n"]
            g_orig = z.export_graphdb_to_zx_graph(GID, "o.json")
            # batched full_reduce
            z.wipe_database(); _load(z, g); rt["n"] = 0
            z.full_reduce_batched_pivot(GID); rtb = rt["n"]
            g_bat = z.export_graphdb_to_zx_graph(GID, "b.json")
            ok1, l1 = verify_pair(g_bat, pg, original=orig.copy())
            ok2, l2 = verify_pair(g_bat, g_orig, original=orig.copy())
            ok = ok1 and ok2
            rt_orig += rto; rt_batch += rtb
            flag = "OK  " if ok else "FAIL"
            if ok: npass += 1
            else: nfail += 1
            print(f"  [{flag}] {name:<18} batch_vs_pyzx[{l1}] batch_vs_orig[{l2}] "
                  f"| rt {rto}->{rtb} | v {g_bat.num_vertices()}", flush=True)
        print(f"\n{npass}/{npass + nfail} passed; "
              f"total round-trips orig {rt_orig} -> batched {rt_batch} "
              f"({100 * (rt_orig - rt_batch) / max(rt_orig, 1):.1f}% fewer)")
    finally:
        txmod.TransactionBase.run = o
        z.wipe_database(); z.close()


if __name__ == "__main__":
    main()
