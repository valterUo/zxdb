import random
import unittest
import pyzx as zx
import json
from utils import benchmark_rule, zx_graph_to_db
from zxdb.zxdb import ZXdb

SEED = 1337
random.seed(SEED)

# python -m unittest tests.test_bialgebra_simp
class TestBialgebraSimp(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()
        self.qubits = 2
        #with open("circuits\\bialgebra_circuit.json", "r") as f:
        #    circuit_json = json.load(f)
        #circuit = zx.Graph().from_json(circuit_json)
        circuit = zx.generate.CNOT_HAD_PHASE_circuit(2, depth = 10*self.qubits)
        self.zx_graph = zx_graph_to_db(self.zxdb, circuit)

    def test_bialgebra_simp(self):
        rule_functions = [self.zxdb.bialgebra_simp, zx.bialg_simp]
        rule_names = ["db_bialgebra_simp", "pyzx_bialgebra_simp"]
        benchmark_rule(rule_functions, 
                       rule_names, 
                       self.zx_graph, 
                       self.zxdb, 
                       self.qubits,
                       test_isomorphism=False,
                       test_degree_distributions=False,
                       visualize=True)

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()