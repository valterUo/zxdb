from fractions import Fraction
import time
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph
import matplotlib
matplotlib.use('Agg')

from zxdb.zxdb import ZXdb

# python -m unittest tests.test_spider_fusion
class TestSpiderFusion(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()
        self.qubits = 1000
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=self.qubits**2,clifford=False)
        
        #c = zx.Circuit(self.qubits)
        #c.add_gate("H", 0)
        #c.add_gate("CNOT", 0, 1) # -> 1
        #c.add_gate("CNOT", 1, 0) # 5 -> 0
        #c.add_gate("H", 2)
        #c.add_gate("CNOT", 2, 1) # 5 -> 1
        #c.add_gate("CNOT", 0, 2)
        #c.add_gate("CNOT", 2, 1)
        #c.add_gate("CNOT", 0, 2) # 6 -> 0
        #c.add_gate("CNOT", 0, 1) # 6 -> 1
        #c.add_gate("XPhase", 0, phase=Fraction(1, 2)) # 6 -> 0
        #c.add_gate("CNOT", 0, 1) # 7 -> 0
        #c.add_gate("CNOT", 0, 1)

        self.zx_graph = circuit_to_graph(c, compress_rows=False)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example1.png")
        

        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,
            batch_size=self.qubits**2
            )

    def test_spider_fusion(self):
        self.zxdb.spider_fusion(graph_id="example_graph")

        graph = self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )

        starttime = time.time()
        return_int = zx.spider_simp(self.zx_graph)
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example2.png")
        endtime = time.time()
        print(f"Time taken for spider fusion with PyZX: {endtime - starttime} seconds with number of {return_int} many simplifications.")
        
        # Check if the graph is empty after Hadamard cancellation
        if self.qubits < 12:
            self.assertTrue(zx.compare_tensors(graph, self.zx_graph), "Spider fusion did not reduce the number of vertices as expected.")
        
        print(graph.stats())

        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Spider fusion did not reduce the number of vertices as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()