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

SEED = 1337

# python -m unittest tests.test_pivot_rule
class TestSpiderFusion(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        #c = cliffordT(3,10,0.1)

        self.zx_graph = cliffordT(4, 13, 0.3)
        # Prepare the ZX graph, follows the structure in PyZX
        zx.spider_simp(self.zx_graph)
        zx.to_gh(self.zx_graph)
        zx.spider_simp(self.zx_graph)
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

        starttime = time.time()
        return_int = zx.pivot_simp(self.zx_graph)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example2.png")
        endtime = time.time()
        print(f"Time taken for pivot simplification with PyZX: {endtime - starttime} seconds with number of {return_int} many simplifications.")
        
        # Check if the graph is empty after Hadamard cancellation
        if  True:
            t1 = tensorfy(graph)
            t2 = tensorfy(self.zx_graph)
            self.assertTrue(compare_tensors(t1, t2), "Pivot rule did not reduce the number of vertices as expected.")
        
        print(graph.stats())

        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Pivot rule did not reduce the number of vertices as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()