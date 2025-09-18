from fractions import Fraction
import random
import time
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph
import matplotlib

from zxdb.pyzx_utils import edge_matcher, node_matcher, pyzx_to_networkx_manual
matplotlib.use('Agg')
from pyzx.generate import cliffordT
from pyzx.tensor import tensorfy, compare_tensors

from zxdb.zxdb import ZXdb
from pyzx.circuit import Circuit
import networkx as nx

SEED = 1337

# python -m unittest tests.test_pivot_rule
class TestPivotRule(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        filepath = "circuits\\pivot_circuit2.json"
        with open(filepath, "r") as f:
            circuit_json = json.load(f)
        self.zx_graph = zx.Graph().from_json(circuit_json)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example1.png")
        
        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,
            batch_size=1000
            )

    def test_pivot_rule(self):

        self.zxdb.pivot_rule(graph_id="example_graph")

        self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="result.json"
        )
        zx.pivot_simp(self.zx_graph)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example2.png")

        filepath = "result.json"
        with open(filepath, "r") as f:
            circuit_json = json.load(f)
        graph = zx.Graph().from_json(circuit_json)

        print(graph.stats())

        # Manually convert both pyzx graphs to networkx graphs
        nxg1 = pyzx_to_networkx_manual(self.zx_graph)
        nxg2 = pyzx_to_networkx_manual(graph)

        print("Graph 1 nodes:", nxg1.nodes(data=True))
        print("Graph 2 nodes:", nxg2.nodes(data=True))

        # Check for isomorphism using the custom property matchers
        are_isomorphic = nx.is_isomorphic(
            nxg1, 
            nxg2, 
            node_match=node_matcher, 
            edge_match=edge_matcher
        )

        nx.draw(nxg1, with_labels=True)
        nx.draw(nxg2, with_labels=True)
        matplotlib.pyplot.savefig("nxg1.png")
        matplotlib.pyplot.savefig("nxg2.png")

        # The main assertion for your test
        self.assertTrue(are_isomorphic, "Graphs are not structurally equivalent (isomorphic) after rule application.")

        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Pivot rule did not reduce the number of vertices as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()