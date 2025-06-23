from fractions import Fraction
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph

from zxdb.zxdb import ZXdb

# python -m unittest tests.test_identity_cancellation
class TestIdentityCancel(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()
        self.qubits = 10
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=10,clifford=False)
        c.add_gate("HAD", 0)
        c.add_gate("ZPhase", 0, Fraction(4, 2))
        c.add_gate("HAD", 0)
        c.add_gate("ZPhase", 0, Fraction(2, 1))
        c.add_gate("ZPhase", 0, Fraction(1, 2))

        self.zx_graph = circuit_to_graph(c, compress_rows=False)
        
        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,  # Clear existing data for this graph_id
            batch_size=1000  # Process in batches of a million gates for better performance
            )

    def test_identity_cancel(self):
        self.zxdb.remove_identities(graph_id="example_graph")

        # Fetch the graph
        graph = self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )
        
        zx.rules.apply_rule(self.zx_graph, zx.rules.remove_ids, zx.rules.match_ids_parallel(self.zx_graph))

        self.assertTrue(zx.compare_tensors(graph, self.zx_graph), "Idendity cancellation did not remove all identities as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()