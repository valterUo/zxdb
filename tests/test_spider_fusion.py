import random
import unittest
import pyzx as zx
from utils import benchmark_rule, zx_graph_to_db
from zxdb.zxdb import ZXdb
SEED = 10
random.seed(SEED)

# python -m unittest tests.test_spider_fusion
class TestSpiderFusion(unittest.TestCase):

    def setUp(self):
        self.zxdb = ZXdb()
        self.qubits = 100
        circuit = zx.generate.CNOT_HAD_PHASE_circuit(qubits=self.qubits,depth=10*self.qubits,clifford=False)
        print("Generated circuit with", self.qubits, "qubits.")
        self.zx_graph = zx_graph_to_db(self.zxdb, circuit)

    def test_spider_fusion(self):
        rule_functions = [self.zxdb.spider_fusion, zx.spider_simp]
        rule_names = ["db_spider_fusion", "pyzx_spider_fusion"]
        benchmark_rule(rule_functions, rule_names, self.zx_graph, self.zxdb, self.qubits)

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()