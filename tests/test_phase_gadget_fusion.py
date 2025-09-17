from fractions import Fraction
import random
import time
import unittest
import pyzx as zx
import json
from pyzx.circuit.graphparser import circuit_to_graph
import matplotlib

from zxdb.pyzx_utils import edge_matcher, node_matcher, phase_poly_term_to_graph, pyzx_to_networkx_manual
matplotlib.use('Agg')
from pyzx.generate import cliffordT, phase_poly_from_gadgets, phase_poly
from pyzx.tensor import tensorfy, compare_tensors

from zxdb.zxdb import ZXdb
from pyzx.circuit import Circuit
import networkx as nx

SEED = 1337

# python -m unittest tests.test_phase_gadget_fusion
class TestPhaseGadgetFusion(unittest.TestCase):

    def setUp(self):
        random.seed(SEED)
        self.circuits = []

        with open("gadget_fusion_hadamard.json", "r") as f:
            circuit_json = json.load(f)
        self.circuits.append(zx.Graph().from_json(circuit_json))

        with open("gadget_fusion_red_green.json", "r") as f:
            circuit_json = json.load(f)
        self.circuits.append(zx.Graph().from_json(circuit_json))

        fig = zx.draw_matplotlib(self.circuits[0])
        fig.savefig("example1.png")

        #self.zx_graph = cliffordT(3,20,0.3)
        #filepath = "example2.json"
        #with open(filepath, "r") as f:
        #    circuit_json = json.load(f)
        #self.zx_graph = zx.Graph().from_json(circuit_json)
        #self.circuits.append(self.zx_graph)
        self.zx_graph = self.circuits[1]
        fig = zx.draw_matplotlib(self.zx_graph)
        fig.savefig("example1.png")
        with open("example.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        self.zxdb = ZXdb()
        
        self.zxdb.import_zx_graph_json_to_graphdb(
            json_file_path="example.json",
            graph_id="example_graph",
            save_metadata=True,
            initialize_empty=True,
            batch_size=1000
            )


    def func_test(self, func, prepare=None):
        for i,c in enumerate(self.circuits):
            with self.subTest(i=i, func=func.__name__):
                if prepare:
                    for f in prepare: f(c,quiet=False)
                t = tensorfy(c)
                fig = zx.draw_matplotlib(c)
                fig.savefig("example2.png")
                func(c, quiet=False)
                fig = zx.draw_matplotlib(c)
                fig.savefig("example3.png")
                t2 = tensorfy(c)
                self.assertTrue(compare_tensors(t,t2))
                del t, t2


    def test_lcomp_rule(self):

        self.func_test(zx.gadget_simp)
        
        self.zxdb.phase_gadget_fusion_rule(graph_id="example_graph")

        self.zxdb.export_graphdb_to_zx_graph(
            graph_id="example_graph",
            json_file_path="result.json"
        )

        filepath = "result.json"
        with open(filepath, "r") as f:
            circuit_json = json.load(f)
        graph = zx.Graph().from_json(circuit_json)

        zx.spider_simp(graph)

        fig = zx.draw_matplotlib(graph)
        fig.savefig("from_graph_db.png")
        
        print(graph.stats())
        print(self.zx_graph.stats())

        with open("result2.json", "w") as f:
            json.dump(json.loads(self.zx_graph.to_json()), f, indent = 4)

        # Manually convert both pyzx graphs to networkx graphs
        nxg1 = pyzx_to_networkx_manual(self.zx_graph)
        nxg2 = pyzx_to_networkx_manual(graph)

        # Check for isomorphism using the custom property matchers
        are_isomorphic = nx.is_isomorphic(
            nxg1, 
            nxg2, 
            node_match=node_matcher, 
            edge_match=edge_matcher
        )

        # The main assertion for your test
        self.assertTrue(are_isomorphic, "Graphs are not structurally equivalent (isomorphic) after rule application.")
        
        # Optional: You can still compare stats as a quick sanity check
        self.assertEqual(graph.stats(), self.zx_graph.stats(), "Graph stats do not match.")

    def tearDown(self):
        self.zxdb.close()

if __name__ == '__main__':
    unittest.main()