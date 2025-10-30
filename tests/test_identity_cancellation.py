import random
import unittest
import pyzx as zx

from utils import benchmark_rule, zx_graph_to_db
from zxdb.zxdb import ZXdb

SEED = 42

# OK
# python -m unittest tests.test_identity_cancellation
class TestIdentityCancel(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.zxdb = ZXdb()
        self.qubits = 1
        #circuit = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=3*self.qubits,clifford=False)

        circuit = zx.circuit.Circuit(self.qubits)
        for _ in range(1000):
            circuit.add_gate("ZPhase", 0, phase = 0)

        self.zx_graph = zx_graph_to_db(self.zxdb, circuit)

    def test_identity_cancel(self):
        rule_functions = [self.zxdb.remove_identities, zx.id_simp]
        rule_names = ["db_identity_cancellation", "pyzx_identity_cancellation"]
        
        #rule_functions = [zx.id_simp]
        #rule_names = ["pyzx_identity_cancellation"]

        benchmark_rule(rule_functions, 
                       rule_names, 
                       self.zx_graph, 
                       self.zxdb, 
                       self.qubits,
                       rule="identity_cancellation", 
                       visualize=False,
                       test_isomorphism=True,
                       test_degree_distributions=True,
                       test_tensor_equivalence=True)

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()