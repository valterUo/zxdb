"""Offline evidence must survive round trips without simplifying the graph."""
import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyzx as zx

from evaluation.evidence import read_graph, save_graph, check_reduction_fixed_point, _ordered_audit_graph
from evaluation.workloads import payload, build


class EvidenceTests(unittest.TestCase):
    def test_audit_order_preserves_exact_graph_and_parallel_edges(self):
        graph = zx.Graph(backend='multigraph')
        graph.set_auto_simplify(False)
        for v in (2, 7, 11, 20):
            graph.add_vertex_indexed(v)
            graph.set_type(v, zx.VertexType.BOUNDARY if v in (2, 20) else zx.VertexType.Z)
        graph.set_phase(7, Fraction(1, 7))
        graph.set_ground(11, True)
        for edge in ((7, 11), (2, 7), (11, 20)):
            graph.add_edge(edge, zx.EdgeType.HADAMARD)
        graph.set_inputs((20,)); graph.set_outputs((2,))
        graph.scalar.floatfactor = 1.0000001
        graph.scalar.add_phase(Fraction(3, 4))
        graph.scalar.add_power(-1234)
        ordered = _ordered_audit_graph(graph)
        self.assertEqual(next(iter(ordered.neighbors(7))), 2)
        self.assertTrue(ordered.is_ground(11))
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            self.assertEqual(save_graph(directory, graph), save_graph(directory, ordered))
        graph.add_edge((7, 11), zx.EdgeType.SIMPLE)
        self.assertIs(_ordered_audit_graph(graph), graph)
        loop = zx.Graph(backend='multigraph')
        loop.set_auto_simplify(False)
        v = loop.add_vertex(zx.VertexType.Z)
        loop.add_edge((v, v), zx.EdgeType.HADAMARD)
        self.assertIs(_ordered_audit_graph(loop), loop)

    def test_followup_reduction_distinguishes_unfinished_and_reduced_graphs(self):
        graph = build(dict(family='identity_chain', size=3, seed=0))
        before = payload(graph)
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            check = check_reduction_fixed_point(graph, 20, directory)
            self.assertEqual(check['status'], 'completed', check)
            self.assertTrue(check['structure_changed'])
            self.assertEqual(payload(graph), before)
            reduced = read_graph(directory, check['output_key'])
            check = check_reduction_fixed_point(reduced, 20)
            self.assertEqual(check['status'], 'completed', check)
            self.assertFalse(check['structure_changed'])
        self.assertEqual(check_reduction_fixed_point(graph, .001)['status'], 'inconclusive')

    def test_exact_round_trip_and_deduplication(self):
        g = zx.Graph(backend='multigraph')
        g.set_auto_simplify(False)
        for v in (2, 7, 11, 20):
            g.add_vertex_indexed(v)
            g.set_type(v, zx.VertexType.BOUNDARY if v in (2, 20) else zx.VertexType.Z)
        g.set_phase(7, Fraction(1, 7))
        for a, b, t in ((2, 7, 1), (7, 11, 1), (7, 11, 2), (7, 11, 2), (11, 20, 1)):
            g.add_edge((a, b), zx.EdgeType(t))
        g.set_inputs((20,)); g.set_outputs((2,))
        g.scalar.add_power(-1234)
        g.scalar.add_phase(Fraction(3, 4))
        g.scalar.add_node(Fraction(1, 3))
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            key = save_graph(directory, g)
            self.assertEqual(key, save_graph(directory, g))
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertEqual(payload(g), payload(read_graph(directory, key)))
            g.scalar.is_zero = True
            zero_key = save_graph(directory, g)
            self.assertNotEqual(key, zero_key)
            self.assertTrue(read_graph(directory, zero_key).scalar.is_zero)

    def test_near_unit_float_factor_is_not_rounded_away(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            g = zx.Graph()
            ordinary = save_graph(directory, g)
            g.scalar.floatfactor = 1.0000001
            changed = save_graph(directory, g)
            self.assertNotEqual(ordinary, changed)
            restored = read_graph(directory, changed)
            self.assertEqual(restored.scalar.floatfactor, g.scalar.floatfactor)
            self.assertEqual(changed, save_graph(directory, restored))

    def test_corruption_and_invalid_keys_are_rejected(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            key = save_graph(directory, zx.Graph())
            path = Path(directory) / (key + '.json')
            data = json.loads(path.read_text())
            data['inputs'] = [42]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                read_graph(directory, key)
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                save_graph(directory, zx.Graph())
            with self.assertRaises(ValueError):
                read_graph(directory, '../outside')
