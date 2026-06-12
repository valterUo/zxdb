"""
Evaluation of the zxdb full_reduce pipeline against pyzx full_reduce.

Builds ~100 small graphs (random CNOT+H+T circuits, Clifford circuits,
T-heavy and H-heavy circuits, phase-gadget graphs and hand-crafted corner
cases), runs both implementations and compares the results by tensor
equivalence (against the ORIGINAL graph, the hard correctness criterion),
graph isomorphism and degree sequences.

Isomorphism is reported but not required: full_reduce applies many rules and
a different (but individually correct) match order can produce structurally
different, semantically equal normal forms.

Usage:
    python -m evaluation.eval_full_reduce              # full suite (~100)
    python -m evaluation.eval_full_reduce --quick      # ~15 graphs
"""
import random
import sys
import time
from functools import partial

import pyzx as zx

from zxdb.zxdb import ZXdb
from zxdb.generate import PHASE_GADGET_GRAPH
from evaluation.harness import run_case, pyzx_fixpoint
from evaluation import cases as C

# pyzx >= 0.10 rewrites match incrementally; iterate full_reduce to a true
# fixpoint so the reference is deterministic across versions.
pyzx_full_reduce = pyzx_fixpoint(lambda g: zx.full_reduce(g, quiet=True))


def random_circuit_graph(seed, qubits, depth, p_had=0.25, p_t=0.3,
                         clifford=False):
    random.seed(seed)
    c = zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=qubits, depth=depth, p_had=p_had, p_t=p_t, clifford=clifford)
    return c.to_graph()


def build_suite(quick=False):
    suite = []
    seed = 1000

    # 1. Random CNOT+H+T circuits over a grid of sizes (the "ordinary" cases).
    for qubits in (2, 3, 4):
        for depth in (6, 10, 14):
            n = (5 if qubits < 4 else 2) if not quick else 1
            for _ in range(n):
                seed += 1
                suite.append((f"random_q{qubits}_d{depth}_s{seed}",
                              partial(random_circuit_graph, seed, qubits, depth)))

    # 2. Clifford-only circuits (S instead of T): exercise the Clifford loop.
    for qubits in (2, 3):
        for depth in (8, 14):
            n = 4 if not quick else 1
            for _ in range(n):
                seed += 1
                suite.append((f"clifford_q{qubits}_d{depth}_s{seed}",
                              partial(random_circuit_graph, seed, qubits, depth,
                                      clifford=True)))

    # 3. T-heavy circuits: many non-Clifford phases -> gadgetization-heavy.
    for _ in range(12 if not quick else 2):
        seed += 1
        suite.append((f"t_heavy_q3_s{seed}",
                      partial(random_circuit_graph, seed, 3, 12,
                              p_had=0.15, p_t=0.5)))

    # 4. H-heavy circuits: many Hadamards -> identity/H-chain heavy.
    for _ in range(10 if not quick else 2):
        seed += 1
        suite.append((f"h_heavy_q3_s{seed}",
                      partial(random_circuit_graph, seed, 3, 12,
                              p_had=0.5, p_t=0.2)))

    # 5. Phase-gadget graphs from the repo generator.
    if not quick:
        for sizes in ([2, 2], [2, 3], [3, 3], [2, 2, 2]):
            label = "x".join(map(str, sizes))
            suite.append((f"gadgets_{label}",
                          partial(PHASE_GADGET_GRAPH, gadget_sizes=sizes)))

    # 6. Corner cases reused from the per-rule evaluation: configurations that
    #    historically broke individual rules, now pushed through the whole
    #    pipeline.
    corner = [
        ("corner_sf_triangle", C.sf_triangle),
        ("corner_sf_hopf_via_fusion", C.sf_hopf_via_fusion),
        ("corner_sf_phase_wrap", C.sf_phase_wrap),
        ("corner_id_chain", C.id_chain),
        ("corner_id_in_triangle", C.id_in_triangle),
        ("corner_id_in_triangle_hh", C.id_in_triangle_hh),
        ("corner_hc_chain4", C.hc_chain4),
        ("corner_lc_basic", C.lc_basic),
        ("corner_lc_toggle_off", C.lc_toggle_off),
        ("corner_lc_two_candidates", C.lc_two_candidates_shared),
        ("corner_pv_mixed_01", C.pv_mixed_01),
        ("corner_pv_shared_neighbor", C.pv_shared_neighbor),
        ("corner_pv_boundary_general", C.pv_boundary_general),
        ("corner_pg_basic", C.pg_basic),
        ("corner_pg_connected_neighbors", C.pg_connected_neighbors),
        ("corner_pb_basic", C.pb_basic),
        ("corner_gf_two_same_targets", C.gf_two_same_targets),
        ("corner_gf_axel_pi", C.gf_axel_pi),
        ("corner_gf_three_gadgets", C.gf_three_gadgets),
        ("corner_supp_basic_type1", C.supp_basic_type1),
        ("corner_copy_a0_mixed", C.copy_a0_mixed),
        ("corner_copy_cascade", C.copy_cascade),
    ]
    suite += corner if not quick else corner[:4]
    return suite


def main(quick=False):
    suite = build_suite(quick)
    print(f"full_reduce evaluation: {len(suite)} graphs")
    zxdb = ZXdb()
    results = []
    t0 = time.time()
    try:
        for i, (name, builder) in enumerate(suite):
            t = time.time()
            res = run_case(zxdb, "full_reduce", name, builder(),
                           db_rule=zxdb.full_reduce,
                           pyzx_rule=pyzx_full_reduce,
                           require_iso=False)
            res["seconds"] = time.time() - t
            results.append(res)
            flag = "PASS" if res["ok"] else "FAIL"
            extra = ""
            if res["error"]:
                extra = " ERROR: " + res["error"].splitlines()[-1]
            else:
                extra = (f" db={res['db_stats']} pyzx={res['pyzx_stats']}"
                         f" sem={res['db_semantic']} iso={res['isomorphic']}"
                         f" degseq={res['degree_seq_match']}")
            print(f"[{flag}] {i + 1:3}/{len(suite)} {name}"
                  f" ({res['seconds']:.1f}s){extra}")
    finally:
        zxdb.close()

    total = time.time() - t0
    done = [r for r in results if not r["error"]]
    n_ok = sum(1 for r in results if r["ok"])
    n_sem = sum(1 for r in done if r["db_semantic"])
    n_iso = sum(1 for r in done if r["isomorphic"])
    n_deg = sum(1 for r in done if r["degree_seq_match"])
    n_err = sum(1 for r in results if r["error"])
    print("\n" + "=" * 70)
    print(f"full_reduce: {n_ok}/{len(results)} passed "
          f"({total:.0f}s total, {total / max(len(results), 1):.1f}s/graph)")
    print(f"  tensor equivalent : {n_sem}/{len(done)}")
    print(f"  isomorphic to pyzx: {n_iso}/{len(done)} (not required)")
    print(f"  degree seq match  : {n_deg}/{len(done)} (not required)")
    if n_err:
        print(f"  errors            : {n_err}")
    for r in results:
        if not r["ok"]:
            print(f"  FAILED: {r['case']}")
    return n_ok == len(results)


if __name__ == "__main__":
    ok = main(quick="--quick" in sys.argv)
    sys.exit(0 if ok else 1)
