import random
import time
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph

from zxdb.zxdb import ZXdb

SEED = 1337

# python -m unittest tests.test_identity_cancellation
class TestIdentityCancel(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        self.qubits = 100
        c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=self.qubits,clifford=False)
        self.zx_graph = circuit_to_graph(c)
        
        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        start_time = time.time()
        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,
            batch_size=self.qubits**2
            )
        end_time = time.time()
        print(f"Time taken to import graph: {end_time - start_time} seconds")

    def test_identity_cancel(self):
        self.zxdb.remove_identities(graph_id="example_graph")

        # Fetch the graph
        graph = self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="example.json"
        )
        
        starttime = time.time()
        return_int = zx.id_simp(self.zx_graph)
        endtime = time.time()
        print(f"Time taken for identity cancellation: {endtime - starttime} seconds with number of {return_int} many simplifications.")

        # PyZX does not consider Hadamard gates as nodes but DB does.
        num_non_had_vertices = len([v for v in graph.vertices() if graph.type(v) != zx.VertexType.H_BOX])
        self.assertTrue(num_non_had_vertices == self.zx_graph.num_vertices(), "Idendity cancellation did not remove all identities as expected.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()