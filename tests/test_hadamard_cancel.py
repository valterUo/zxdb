import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph

from zxdb.zxdb import ZXdb

# python -m unittest tests.test_hadamard_cancel
class TestHadamardCancel(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()
        self.qubits = 1000
        self.non_hadamard_nodes = 2*self.qubits # Input and output nodes
        c = zx.Circuit(self.qubits)

        # Create a simple circuit with cancelling Hadamard gates
        for i in range(self.qubits):
            # This Hadamard gate will be cancelled
            c.add_gate("HAD", i)
            c.add_gate("HAD", i)
            c.add_gate("CNOT", i, (i + 1) % self.qubits)
            self.non_hadamard_nodes += 2  # CNOT adds two nodes
            # These two Hadamard gates will cancel each other out
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
        self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )

        with open("example.json", "r") as f:
            graph = zx.Graph.from_json(json.load(f))
        
        # Check if the graph is empty after Hadamard cancellation
        self.assertEqual(graph.num_vertices(), self.non_hadamard_nodes, "Hadamard cancellation did not remove all Hadamards as expected.")

    def tearDown(self):
        # Clean up any test data if necessary
        pass

if __name__ == '__main__':
    unittest.main()