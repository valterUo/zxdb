import random
import unittest

import pyzx as zx

from zxdb.zxdb import ZXdb
from evaluation.harness import run_case, pyzx_fixpoint
from evaluation import cases as C

_pyzx_full_reduce = pyzx_fixpoint(lambda g: zx.full_reduce(g, quiet=True))


def _random_circuit_graph(seed, qubits, depth, p_had=0.25, p_t=0.3,
                          clifford=False):
    random.seed(seed)
    c = zx.generate.CNOT_HAD_PHASE_circuit(
        qubits=qubits, depth=depth, p_had=p_had, p_t=p_t, clifford=clifford)
    return c.to_graph()


# python -m unittest tests.test_full_reduce
#
# Fast representative subset of the full_reduce evaluation. The complete
# ~100-graph run lives in `python -m evaluation.eval_full_reduce`.
class TestFullReduce(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()

    def tearDown(self):
        self.zxdb.close()

    def _check(self, name, g):
        res = run_case(
            self.zxdb, "full_reduce", name, g,
            db_rule=self.zxdb.full_reduce,
            pyzx_rule=_pyzx_full_reduce,
            require_iso=False)
        self.assertIsNone(res["error"], msg=f"{name}: {res['error']}")
        # The harness verdict is tiered (tensor -> sampled tensor -> structural)
        # and `level` records which check decided it; for these small graphs it
        # is always tensor-level. ok must hold and the level must not be a
        # tensor-level mismatch.
        self.assertTrue(
            res["ok"],
            msg=f"{name}: DB full_reduce result not verified equivalent to "
                f"pyzx (level={res.get('level')}, db={res['db_stats']}, "
                f"pyzx={res['pyzx_stats']})")
        self.assertIn("tensor", str(res.get("level")),
                      msg=f"{name}: expected tensor-level verification, got "
                          f"{res.get('level')}")
        self.assertEqual(res["parallel_edges"], 0, msg=name)
        self.assertEqual(res["self_loops"], 0, msg=name)

    def test_random_circuits(self):
        for seed, qubits, depth in [(1, 2, 8), (2, 3, 10), (3, 3, 14),
                                    (4, 4, 10)]:
            with self.subTest(seed=seed, qubits=qubits, depth=depth):
                self._check(f"random_q{qubits}_d{depth}_s{seed}",
                            _random_circuit_graph(seed, qubits, depth))

    def test_clifford_circuits(self):
        for seed in (5, 6):
            with self.subTest(seed=seed):
                self._check(f"clifford_s{seed}",
                            _random_circuit_graph(seed, 3, 12, clifford=True))

    def test_t_heavy_circuit(self):
        self._check("t_heavy",
                    _random_circuit_graph(7, 3, 12, p_had=0.15, p_t=0.5))

    def test_corner_cases(self):
        for name, builder in [
            ("id_in_triangle", C.id_in_triangle),
            ("pivot_gadget_basic", C.pg_basic),
            ("gadget_fusion_axel_pi", C.gf_axel_pi),
            ("pivot_boundary_general", C.pv_boundary_general),
        ]:
            with self.subTest(case=name):
                self._check(name, builder())


if __name__ == "__main__":
    unittest.main()
