import unittest
from fractions import Fraction
import pyzx as zx
from utils import benchmark_rule, zx_graph_to_db, pyzx_fixpoint
from zxdb.zxdb import ZXdb

# python -m unittest tests.test_supplementarity
class TestSupplementaritySimp(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()

        # Build a graph-like ZX diagram with two pairs of supplementary phase gadgets.
        # Each pair: a leaf T (phase 1/4) and a leaf T† (phase 3/4) both connected
        # via Hadamard edges to the same center Z-spider.
        # Supplementarity removes both leaves and adds pi to the center's phase.
        g = zx.Graph()

        b_in1 = g.add_vertex(zx.VertexType.BOUNDARY, 0, 0)
        b_in2 = g.add_vertex(zx.VertexType.BOUNDARY, 1, 0)

        z1 = g.add_vertex(zx.VertexType.Z, 0, 1, Fraction(0))
        z2 = g.add_vertex(zx.VertexType.Z, 1, 1, Fraction(0))

        t1 = g.add_vertex(zx.VertexType.Z, 0, 2, Fraction(1, 4))
        t2 = g.add_vertex(zx.VertexType.Z, 0, 3, Fraction(3, 4))
        t3 = g.add_vertex(zx.VertexType.Z, 1, 2, Fraction(1, 4))
        t4 = g.add_vertex(zx.VertexType.Z, 1, 3, Fraction(3, 4))

        b_out1 = g.add_vertex(zx.VertexType.BOUNDARY, 0, 5)
        b_out2 = g.add_vertex(zx.VertexType.BOUNDARY, 1, 5)

        g.add_edge((b_in1, z1), zx.EdgeType.SIMPLE)
        g.add_edge((b_in2, z2), zx.EdgeType.SIMPLE)
        g.add_edge((z1, b_out1), zx.EdgeType.SIMPLE)
        g.add_edge((z2, b_out2), zx.EdgeType.SIMPLE)

        g.add_edge((z1, t1), zx.EdgeType.HADAMARD)
        g.add_edge((z1, t2), zx.EdgeType.HADAMARD)
        g.add_edge((z2, t3), zx.EdgeType.HADAMARD)
        g.add_edge((z2, t4), zx.EdgeType.HADAMARD)

        g.set_inputs((b_in1, b_in2))
        g.set_outputs((b_out1, b_out2))

        self.qubits = 2
        self.zx_graph = zx_graph_to_db(self.zxdb, g)

    def test_supplementarity_simp(self):
        rule_functions = [self.zxdb.supplementarity_simp, pyzx_fixpoint(zx.supplementarity_simp)]
        rule_names = ["db_supplementarity_simp", "pyzx_supplementarity_simp"]
        benchmark_rule(
            rule_functions,
            rule_names,
            self.zx_graph,
            self.zxdb,
            self.qubits,
            rule="supplementarity_simp",
            test_isomorphism=True,
            test_degree_distributions=True,
            test_tensor_equivalence=True,
            visualize=False,
        )

    def tearDown(self):
        self.zxdb.close()


if __name__ == "__main__":
    unittest.main()
