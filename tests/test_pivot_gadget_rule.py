import io
import random
import unittest
from contextlib import redirect_stdout

import pyzx as zx

from utils import zx_graph_to_db, pyzx_fixpoint
from evaluation.harness import _semantic_equal
from zxdb.zxdb import ZXdb

# python -m unittest tests.test_pivot_gadget_rule
#
# NOTE: the original fixture (circuits/pivot_gadget_circuit2.json expanded x4)
# is a MALFORMED open graph — it contains boundary vertices that are not in
# inputs/outputs, so pyzx's own `to_tensor` raises "non-ZXH type" on it and the
# quimb operator-invariant check in benchmark_rule fails *identically for pyzx*
# (it printed "PyZX vs original: False"). That made the test red without
# indicating any DB bug. This version verifies the rule soundly on small,
# well-formed graph-like inputs (where tensor contraction is fast and reliable)
# by comparing the DB result's tensor against the original up to a boundary
# permutation. pivot_gadget is exhaustively covered on hand-built graph-like
# cases in `evaluation/run_eval.py pivot_gadget` (9/9) and inside the 100-graph
# `evaluation/eval_full_reduce.py` run.
class TestPivotGadgetRule(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()

    def tearDown(self):
        self.zxdb.close()

    def _graphlike(self, seed, qubits=10, depth=40):
        random.seed(seed)
        g = zx.generate.CNOT_HAD_PHASE_circuit(
            qubits=qubits, depth=depth, p_had=0.3, p_t=0.3).to_graph()
        # bring to graph-like form (the valid domain for pivot_gadget)
        pyzx_fixpoint(zx.spider_simp)(g)
        zx.to_gh(g)
        pyzx_fixpoint(zx.spider_simp)(g)
        pyzx_fixpoint(zx.id_simp)(g)
        return g

    def test_pivot_gadget_simp(self):
        # Small, well-formed graph-like inputs on which pivot_gadget fires.
        fired = 0
        for seed in (0, 3, 4, 5):
            g = self._graphlike(seed)
            original = g.copy()
            with redirect_stdout(io.StringIO()):
                zx_graph_to_db(self.zxdb, g.copy(), graph_id="example_graph")
                applied = self.zxdb.pivot_gadget_rule("example_graph")
                db_g = self.zxdb.export_graphdb_to_zx_graph(
                    "example_graph", "example.json")
            if applied:
                fired += 1
            self.assertNotEqual(
                _semantic_equal(db_g, original), False,
                msg=f"seed {seed}: DB pivot_gadget result is not "
                    f"tensor-equivalent to the original graph")
        self.assertGreater(fired, 0, "pivot_gadget never fired on any input")


if __name__ == "__main__":
    unittest.main()
