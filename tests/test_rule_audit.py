"""Rule audit: independent dense tensors, adversarial contexts, and no-op guards.

Run against a dedicated Memgraph with ZXDB_LIVE=1. The fixture/reference checks
also run offline; they do not claim to execute the database implementation.
"""
import os
import unittest
from fractions import Fraction
from itertools import product

import numpy as np
import pyzx as zx

from evaluation.cases import RULES, _bialg_base, supp_two_neighbor_pair
from evaluation.core import load, read, validate_graph
from evaluation.workloads import payload
from tests.live_support import ResearchLiveTests
from tests.test_hopf_normalization import dense_tensor

B, Z, X = zx.VertexType.BOUNDARY, zx.VertexType.Z, zx.VertexType.X
S, H = zx.EdgeType.SIMPLE, zx.EdgeType.HADAMARD


def adversarial_context(g):
    """Reverse allocation/edge direction, conjugate phases, and permute ports.

    Boundary Hadamards exercise rewiring without changing interior eligibility.
    The transformed graph is its own semantic baseline, not compared to g.
    """
    h = zx.Graph()
    ids = {v: h.add_vertex(g.type(v), phase=-g.phase(v))
           for v in reversed(list(g.vertices()))}
    for e in reversed(list(g.edges())):
        a, b = g.edge_st(e)
        t = g.edge_type(e)
        if (g.type(a) == B) != (g.type(b) == B):
            t = zx.EdgeType(3 - int(t))
        h.add_edge((ids[b], ids[a]), t)
    h.set_inputs(tuple(ids[v] for v in reversed(g.inputs())))
    h.set_outputs(tuple(ids[v] for v in reversed(g.outputs())))
    h.scalar = g.scalar.copy(conjugate=True)
    return h


def assert_projective(test, original, result):
    """Independent PyZX contraction, including zero/nonzero discrimination."""
    validate_graph(original)
    validate_graph(result)
    test.assertEqual((len(original.inputs()), len(original.outputs())),
                     (len(result.inputs()), len(result.outputs())))
    a, b = dense_tensor(original).ravel(), dense_tensor(result).ravel()
    test.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
    ma, mb = np.max(np.abs(a)), np.max(np.abs(b))
    # These small rational-phase fixtures have no tiny nonzero scalar factors.
    if ma < 1e-10 or mb < 1e-10:
        test.assertLess(ma, 1e-10, 'nonzero input became zero')
        test.assertLess(mb, 1e-10, 'zero input became nonzero')
        return
    a, b = a / ma, b / mb
    scale = np.vdot(a, b) / np.vdot(a, a)
    test.assertGreater(abs(scale), 1e-10)
    np.testing.assert_allclose(b, scale * a, atol=1e-8, rtol=1e-8)


def copy_context(color, phase, edge_type):
    g = zx.Graph()
    leaf = g.add_vertex(color, phase=phase)
    hub = g.add_vertex(color if edge_type == H else Z if color == X else X,
                       phase=Fraction(3, 7))
    g.add_edge((leaf, hub), edge_type)
    ports = []
    for i, t in enumerate((S, H, H)):
        n = g.add_vertex(Z if i % 2 else X, phase=Fraction(i + 1, 5))
        b = g.add_vertex(B)
        g.add_edge((hub, n), t)
        g.add_edge((n, b), H if i % 2 else S)
        ports.append(b)
    # An existing edge between neighbors must survive the hub's deletion.
    g.add_edge((2, 4), H)
    g.set_inputs(tuple(ports[:2]))
    g.set_outputs(tuple(ports[2:]))
    return g


def bialgebra_context(pa, pb, edge_type):
    g, a, b, left, right = _bialg_base(2, 2, v0_phase=pa, v1_phase=pb)
    g.set_edge_type(g.edge(a, left[0]), edge_type)
    g.set_edge_type(g.edge(b, right[-1]), H)
    g.add_edge((left[0], right[0]), S)
    g.add_edge((left[-1], right[-1]), H)
    return g


class AuditFixtureTests(unittest.TestCase):
    def test_every_rule_reference_in_adversarial_context(self):
        for rule, spec in RULES.items():
            for name, builder, *_ in spec['cases']:
                with self.subTest(rule=rule, case=name):
                    g = adversarial_context(builder())
                    result = g.copy()
                    spec['pyzx'](result)
                    assert_projective(self, g, result)

    def test_copy_and_bialgebra_parameter_matrix_reference(self):
        for color, phase, edge in product((Z, X), (0, 1), (S, H)):
            g = copy_context(color, phase, edge)
            h = g.copy()
            RULES['copy']['pyzx'](h)
            self.assertNotEqual(payload(g), payload(h))
            assert_projective(self, g, h)
        for pa, pb, edge in product((0, 1), (0, 1), (S, H)):
            g = bialgebra_context(pa, pb, edge)
            h = g.copy()
            RULES['bialgebra']['pyzx'](h)
            self.assertNotEqual(payload(g), payload(h))
            assert_projective(self, g, h)


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1', 'Opt-in Memgraph checks')
class RuleAuditTests(ResearchLiveTests):
    def apply(self, rule, variant):
        method = RULES[rule]['db_method']
        if variant == 'optimized':
            return self.db.apply_rule_optimized(method, self.gid)
        return getattr(self.db, method)(self.gid)

    def check(self, g, rule, variant, must_fire=False, must_not_fire=False):
        load(self.db, g, self.gid)
        before = payload(read(self.db, self.gid))
        count = self.apply(rule, variant)
        result = read(self.db, self.gid)
        if must_fire:
            self.assertGreater(count, 0)
            self.assertNotEqual(before, payload(result))
        if must_not_fire:
            self.assertEqual(count, 0)
            self.assertEqual(before, payload(result))
        assert_projective(self, g, result)
        # Hadamard cancellation deliberately processes a single labeling pass.
        if rule != 'hadamard_cancel':
            self.assertEqual(self.apply(rule, variant), 0)
            self.assertEqual(payload(read(self.db, self.gid)), payload(result))

    def test_all_eleven_rules_in_reordered_conjugated_contexts(self):
        for rule, spec in RULES.items():
            for name, builder, *_ in spec['cases']:
                for variant in (('control',) if rule == 'hadamard_cancel'
                                else ('control', 'optimized')):
                    with self.subTest(rule=rule, case=name, variant=variant):
                        self.check(adversarial_context(builder()), rule, variant,
                                   must_not_fire=name.startswith('no_'))

    def test_copy_colors_phases_and_mixed_incident_wires(self):
        for color, phase, edge, variant in product((Z, X), (0, 1), (S, H),
                                                  ('control', 'optimized')):
            with self.subTest(color=color, phase=phase, edge=edge, variant=variant):
                self.check(copy_context(color, phase, edge), 'copy', variant, True)

    def test_bialgebra_both_pauli_phases_and_existing_neighbor_edges(self):
        for pa, pb, edge, variant in product((0, 1), (0, 1), (S, H),
                                            ('control', 'optimized')):
            with self.subTest(pa=pa, pb=pb, edge=edge, variant=variant):
                self.check(bialgebra_context(pa, pb, edge), 'bialgebra', variant, True)

    def test_gadget_fusion_distinguishes_target_wire_types(self):
        for variant, same in product(('control', 'optimized'), (False, True)):
            with self.subTest(variant=variant, same=same):
                g = zx.Graph()
                targets = [g.add_vertex(Z, phase=Fraction(1, 3)),
                           g.add_vertex(Z, phase=Fraction(2, 5))]
                ports = []
                for target in targets:
                    port = g.add_vertex(B)
                    g.add_edge((target, port), S)
                    ports.append(port)
                for i in range(2):
                    center = g.add_vertex(Z)
                    tip = g.add_vertex(Z, phase=Fraction(1 + 2 * i, 4))
                    g.add_edge((center, tip), H)
                    g.add_edge((center, targets[0]), H if same or i == 0 else S)
                    g.add_edge((center, targets[1]), H)
                g.set_inputs((ports[0],))
                g.set_outputs((ports[1],))
                load(self.db, g, self.gid)
                before = payload(read(self.db, self.gid))
                count = self.apply('gadget_fusion', variant)
                result = read(self.db, self.gid)
                assert_projective(self, g, result)
                if same:
                    self.assertGreater(count, 0)
                else:
                    self.assertEqual(count, 0)
                    self.assertEqual(payload(result), before)

    def test_spider_fusion_preserves_hadamard_self_loop_phase(self):
        for color, edge in product((Z, X), (S, H)):
            with self.subTest(color=color, edge=edge):
                g = zx.Graph(backend='multigraph')
                g.set_auto_simplify(False)
                a = g.add_vertex(color, phase=Fraction(1, 4))
                b = g.add_vertex(color, phase=Fraction(1, 3))
                bi, bo = g.add_vertex(B), g.add_vertex(B)
                g.add_edge((bi, a), S)
                g.add_edge((a, b), S)
                g.add_edge((b, bo), S)
                g.add_edge((a, a), edge)
                g.set_inputs((bi,))
                g.set_outputs((bo,))
                self.check(g, 'spider_fusion', 'control', True)

    def test_single_gadget_normalization_reports_progress(self):
        from evaluation.cases import gf_single_axel_pi
        for variant in ('control', 'optimized'):
            with self.subTest(variant=variant):
                self.check(gf_single_axel_pi(), 'gadget_fusion', variant, True)

    def test_identity_hidden_by_a_simple_self_loop(self):
        for color, variant in product((Z, X), ('control', 'optimized')):
            with self.subTest(color=color, variant=variant):
                g = zx.Graph(backend='multigraph')
                g.set_auto_simplify(False)
                bi, bo, v = g.add_vertex(B), g.add_vertex(B), g.add_vertex(color)
                g.add_edge((bi, v), S)
                g.add_edge((v, bo), H)
                g.add_edge((v, v), S)
                g.set_inputs((bi,))
                g.set_outputs((bo,))
                self.check(g, 'identity', variant, True)
                result = read(self.db, self.gid)
                self.assertEqual(result.num_vertices(), 2)
                self.assertEqual(result.num_edges(), 1)

    def test_legacy_bialgebra_clique_requires_zero_phases_and_simple_wires(self):
        # Retained query entries are callable even though bialgebra_simp uses
        # the newer two-center implementation.
        for invalid in (None, 'phase', 'hadamard'):
            with self.subTest(invalid=invalid):
                g = zx.Graph()
                left = [g.add_vertex(Z) for _ in range(2)]
                right = [g.add_vertex(X) for _ in range(2)]
                for a, b in product(left, right):
                    g.add_edge((a, b), S)
                ports = []
                for v in left + right:
                    port = g.add_vertex(B)
                    g.add_edge((v, port), S)
                    ports.append(port)
                g.set_inputs(tuple(ports[:2]))
                g.set_outputs(tuple(ports[2:]))
                if invalid == 'phase':
                    g.set_phase(left[0], Fraction(1, 4))
                elif invalid == 'hadamard':
                    g.set_edge_type(g.edge(left[0], ports[0]), H)
                load(self.db, g, self.gid)
                before = payload(read(self.db, self.gid))
                with self.db.driver.session() as session:
                    session.run(self.db._query('Bialgebra labeling'), graph_id=self.gid).consume()
                    session.run(self.db._query('Bialgebra simplification'), graph_id=self.gid).consume()
                result = read(self.db, self.gid)
                assert_projective(self, g, result)
                if invalid:
                    self.assertEqual(payload(result), before)
                else:
                    self.assertLess(result.num_vertices(), g.num_vertices())

    def test_supplementarity_rejects_non_graphlike_neighborhoods(self):
        for adjacent, invalid, variant in product((False, True),
                ('simple', 'mixed', 'x_neighbor', 'self_loop', 'parallel'),
                ('control', 'optimized')):
            with self.subTest(adjacent=adjacent, invalid=invalid, variant=variant):
                g = supp_two_neighbor_pair()
                tips = [v for v in g.vertices() if g.type(v) == Z and g.vertex_degree(v) == 2]
                a, b = tips
                if adjacent:
                    g.set_phase(b, Fraction(7, 4))
                    g.add_edge((a, b), H)
                if invalid in ('simple', 'mixed'):
                    for e in list(g.incident_edges(a)):
                        g.set_edge_type(e, S)
                    if invalid == 'simple':
                        for e in list(g.incident_edges(b)):
                            g.set_edge_type(e, S)
                elif invalid == 'x_neighbor':
                    g.set_type(next(n for n in g.neighbors(a) if n != b), X)
                else:
                    # Preserve multiedges and loops, which GraphS simplifies.
                    source = g
                    g = zx.Graph(backend='multigraph')
                    g.set_auto_simplify(False)
                    for v in source.vertices():
                        self.assertEqual(g.add_vertex(source.type(v), phase=source.phase(v)), v)
                    for e in source.edges():
                        g.add_edge(source.edge_st(e), source.edge_type(e))
                    g.set_inputs(source.inputs())
                    g.set_outputs(source.outputs())
                    g.add_edge((a, a) if invalid == 'self_loop'
                               else (a, next(n for n in g.neighbors(a) if n != b)), H)
                load(self.db, g, self.gid)
                before = payload(read(self.db, self.gid))
                self.assertEqual(self.apply('supplementarity', variant), 0)
                self.assertEqual(payload(read(self.db, self.gid)), before)


if __name__ == '__main__':
    unittest.main()
