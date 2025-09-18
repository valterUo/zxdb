from fractions import Fraction
import random
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph
import matplotlib

from zxdb.pyzx_utils import edge_matcher, node_matcher, phase_poly_term_to_graph, pyzx_to_networkx_manual
matplotlib.use('Agg')
from pyzx.tensor import tensorfy, compare_tensors

from zxdb.zxdb import ZXdb
import networkx as nx

SEED = 1337

# python -m unittest tests.test_pivot_boundary_rule
class TestPivotBoundaryRule(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.circuits = []
        with open("circuits\\pivot_boundary_circuit2.json", "r") as f:
            circuit_json = json.load(f)

        self.circuits.append(zx.Graph(backend = 'multigraph').from_json(circuit_json))

        fig = zx.draw_matplotlib(self.circuits[0])
        fig.savefig("example1.png")

        self.zx_graph = self.circuits[0]
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example1.png")
        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb = ZXdb()
        
        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,
            batch_size=1000
            )

    def test_pivot_boundary_simp(self):
        
        zx.pivot_boundary_simp(self.zx_graph)

        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example2.png")
        
        
        #self.zxdb.phase_gadget_fusion_rule(graph_id="example_graph")

        #self.zxdb.export_graphdb_to_zx_graph(
        #    graph_id="example_graph",
        #    json_file_path="result.json"
        #)

        filepath = "result.json"
        with open(filepath, "r") as f:
            circuit_json = json.load(f)
        graph = zx.Graph().from_json(circuit_json)

        #zx.spider_simp(graph)

        fig = zx.draw_matplotlib(graph)
        fig.savefig("from_graph_db.png")
        
        print(graph.stats())
        print(self.zx_graph.stats())

        with open("result2.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        # Manually convert both pyzx graphs to networkx graphs
        nxg1 = pyzx_to_networkx_manual(self.zx_graph)
        nxg2 = pyzx_to_networkx_manual(graph)

        # Check for isomorphism using the custom property matchers
        are_isomorphic = nx.is_isomorphic(
            nxg1, 
            nxg2, 
            node_match=node_matcher, 
            edge_match=edge_matcher
        )

        # The main assertion for your test
        self.assertTrue(are_isomorphic, "Graphs are not structurally equivalent (isomorphic) after rule application.")
        
        # Optional: You can still compare stats as a quick sanity check
        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Graph stats do not match.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()