from random import random
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph

from zxdb.zxdb import ZXdb

SEED = 1337

# python -m unittest tests.test_hadamard_cancellation
class TestHadamardCancellation(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        self.qubits = 10
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=1000,clifford=False)

        # Create a simple circuit with cancelling Hadamard gates
        for i in range(self.qubits):
            # This Hadamard gate will be cancelled
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)
            c.add_gate("CNOT", i, (i + 1) % self.qubits)
            # Four of these five Hadamard gates will cancel each other out
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)

        self.zx_graph = circuit_to_graph(c, compress_rows=False)

        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,  # Clear existing data for this graph_id
            batch_size=10000  # Process in batches of a million gates for better performance
            )

    def test_Hadamard_cancel(self):
        # Test the Hadamard cancellation
        self.zxdb.hadamard_cancel(graph_id="example_graph")
        

        # Fetch the graph after Hadamard cancellation
        graph = self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )
        
        # Check if the graph is empty after Hadamard cancellation
        self.assertTrue(zx.compare_tensors(graph, self.zx_graph), "Hadamard cancellation did not remove all Hadamards as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()