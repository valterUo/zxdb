from fractions import Fraction
import random
import time
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph
import matplotlib
matplotlib.use('Agg')
from pyzx.generate import cliffordT
from pyzx.tensor import tensorfy, compare_tensors

from zxdb.zxdb import ZXdb
from pyzx.circuit import Circuit

SEED = 1337

# python -m unittest tests.test_pivot_rule
class TestPivotRule(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        filepath = "pivot_circuit.json"
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

        graph = self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )

        zx.pivot_simp(self.zx_graph)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example2.png")
        
        print(graph.stats())

        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Pivot rule did not reduce the number of vertices as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()