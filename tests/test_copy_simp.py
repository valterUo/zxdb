import unittest
from fractions import Fraction
from pyzx.simplify import copy_simp
import pyzx as zx
from utils import benchmark_rule, zx_graph_to_db, pyzx_fixpoint
from zxdb.zxdb import ZXdb


# python -m unittest tests.test_copy_simp
class TestCopySimp(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()

        # Build a graph-like ZX diagram with one copy leaf (v, phase=0) connected
        # to a Z-spider w that has boundary neighbors (b_in, b_out) and two interior
        # Z-neighbors (n1, n2).  After copy_simp: v and w are deleted; n1 and n2
        # become isolated and are removed; two new Z(0) leaves are created connected
        # to b_in and b_out via Hadamard edges.
        g = zx.Graph()

        b_in = g.add_vertex(zx.VertexType.BOUNDARY, 0, 0)
        b_out = g.add_vertex(zx.VertexType.BOUNDARY, 0, 4)

        w = g.add_vertex(zx.VertexType.Z, 0, 1, Fraction(1, 4))
        v = g.add_vertex(zx.VertexType.Z, 0, 2, Fraction(0))    # copy leaf
        n1 = g.add_vertex(zx.VertexType.Z, 0, 3, Fraction(1, 4))
        n2 = g.add_vertex(zx.VertexType.Z, 0, 4, Fraction(3, 4))

        g.add_edge((b_in, w), zx.EdgeType.SIMPLE)
        g.add_edge((w, b_out), zx.EdgeType.SIMPLE)
        g.add_edge((w, v), zx.EdgeType.HADAMARD)
        g.add_edge((w, n1), zx.EdgeType.HADAMARD)
        g.add_edge((w, n2), zx.EdgeType.HADAMARD)

        g.set_inputs((b_in,))
        g.set_outputs((b_out,))

        self.qubits = 1
        self.zx_graph = zx_graph_to_db(self.zxdb, g)

    def test_copy_simp(self):
        rule_functions = [self.zxdb.copy_simp, pyzx_fixpoint(copy_simp)]
        rule_names = ["db_copy_simp", "pyzx_copy_simp"]
        benchmark_rule(
            rule_functions,
            rule_names,
            self.zx_graph,
            self.zxdb,
            self.qubits,
            rule="copy_simp",
            test_isomorphism=True,
            test_degree_distributions=True,
            test_tensor_equivalence=False,
            visualize=False,
        )

    def tearDown(self):
        self.zxdb.close()


if __name__ == "__main__":
    unittest.main()
