"""Optimized reduction regressions; live tests require an empty dedicated server."""
import os
import time
import unittest
from unittest.mock import patch
from itertools import product
from itertools import combinations
from fractions import Fraction

import pyzx as zx

from evaluation.core import load, read, verify, reference, compare_outputs, CountingDriver
from evaluation.workloads import build, metrics, payload
from tests.test_pivot_materialization import pivot_neighborhood
from tests.live_support import ResearchLiveTests


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1', 'Opt-in Memgraph checks')
class ReductionOptimizationTests(unittest.TestCase):
    setUp = ResearchLiveTests.setUp
    tearDown = ResearchLiveTests.tearDown

    def test_color_change_repeated_loads_and_exact_wire_parity(self):
        for seed in range(6):
            g = build(dict(family='random_clifford_t', size=3, depth=36, seed=seed))
            expected = g.copy()
            count = sum(g.type(v) == zx.VertexType.X for v in g.vertices())
            for edge in expected.edges():
                a, b = expected.edge_st(edge)
                if (int(g.type(a) == zx.VertexType.X) + int(g.type(b) == zx.VertexType.X)) % 2:
                    expected.set_edge_type(edge, zx.EdgeType(3 - int(expected.edge_type(edge))))
            for v in expected.vertices():
                if expected.type(v) == zx.VertexType.X: expected.set_type(v, zx.VertexType.Z)
            load(self.db, g, self.gid)
            self.assertEqual(self.db.to_gh(self.gid), count)
            result = read(self.db, self.gid)
            a, b = payload(expected), payload(result)
            a.pop('scalar'); b.pop('scalar')
            self.assertEqual(a, b)
            self.assertEqual(verify(g, result)['status'], 'verified')
            self.assertEqual(self.db.to_gh(self.gid), 0)

    def test_exclusive_scope_checks_ownership_once_and_resets_after_error(self):
        graph = build(dict(family='identity', size=2, seed=0))
        load(self.db, graph, self.gid)
        driver = CountingDriver(self.db.driver)
        self.db._driver = driver
        try:
            with self.db._optimized_scope(self.gid, exclusive_access=True):
                with self.db.driver.session() as session:
                    for _ in range(4):
                        self.db._assert_graph_isolated(session, self.gid)
                self.assertEqual(driver.submissions, 1)
            with self.assertRaises(RuntimeError):
                with patch.object(self.db, '_full_reduce_body', side_effect=RuntimeError('Injected failure')):
                    self.db.full_reduce_optimized(self.gid, exclusive_access=True)
            self.assertIsNone(self.db._reduction_options.get())
            count = driver.submissions
            with self.db.driver.session() as session:
                self.db._assert_graph_isolated(session, self.gid)
            self.assertEqual(driver.submissions, count+1)
        finally:
            self.db._driver = driver.driver

    def test_exclusive_full_reduce_rejects_cross_graph_edge_before_mutation(self):
        graph = build(dict(family='identity', size=1, seed=0))
        load(self.db, graph, self.gid)
        foreign = self.gid + '_foreign'
        try:
            with self.db.driver.session() as s:
                s.run('MATCH(n:Node {graph_id:$gid}) WITH n LIMIT 1 '
                      'CREATE(n)-[:Wire {t:1}]->(:Node {graph_id:$foreign,t:1,phase:0})',
                      gid=self.gid, foreign=foreign).consume()
            with self.assertRaises(ValueError):
                self.db.full_reduce_optimized(self.gid, exclusive_access=True)
            self.assertIsNone(self.db._reduction_options.get())
            with self.db.driver.session() as s:
                self.assertEqual(s.run('MATCH(n:Node {graph_id:$gid}) RETURN count(n) AS n',
                                       gid=self.gid).single()['n'], graph.num_vertices())
        finally:
            self.db.empty_graphdb(foreign)

    def test_parallel_invariant_skips_noops_but_invalidates_on_mutation(self):
        from evaluation.cases import sf_hadamard_parallel_via_fusion, id_in_triangle_hh
        graph = build(dict(family='lcomp', size=2, seed=0))
        load(self.db, graph, self.gid)
        with self.db._optimized_scope(self.gid, exclusive_access=True):
            with self.db.driver.session() as s:
                self.db._normalize_parallel_edges_tx(s, self.gid)
                with patch.object(self.db, '_run_count', wraps=self.db._run_count) as count:
                    self.db._normalize_parallel_edges_tx(s, self.gid)
                    self.assertEqual(count.call_count, 0)
            self.db.local_complementation_rule(self.gid)
            self.assertFalse(self.db._parallel_state(self.gid)['parallel_normalized'])
        for builder in (sf_hadamard_parallel_via_fusion, id_in_triangle_hh):
            graph = builder()
            load(self.db, graph, self.gid)
            self.db.full_reduce_optimized(self.gid, exclusive_access=True)
            result = read(self.db, self.gid)
            self.assertEqual(verify(graph, result)['status'], 'verified')
        self.assertIsNone(self.db._parallel_state(self.gid))

    def test_limited_guards_keep_zero_one_and_multiple_match_behavior(self):
        for size in (0, 1, 8):
            graph = build(dict(family='spider_fusion', size=size, seed=0))
            load(self.db, graph, self.gid)
            with self.db._optimized_scope(self.gid):
                with patch.object(self.db, '_run_count', wraps=self.db._run_count) as count:
                    self.assertEqual(self.db.spider_fusion(self.gid), size)
                    self.db.supplementarity_simp(self.gid)
                    queries = [call.args[1] for call in count.call_args_list]
                    self.assertTrue(any('WITH r LIMIT 1' in q for q in queries))
                    self.assertTrue(any('WITH v LIMIT 2' in q for q in queries))


    def test_repeated_boundary_load_preserves_graph_and_order(self):
        from evaluation.workloads import payload
        graph = build(dict(family='copies_lcomp', size=128, seed=0))
        expected = payload(graph)
        for attempt in range(80):
            load(self.db, graph, self.gid)
            self.assertEqual(payload(read(self.db, self.gid)), expected, attempt)

    def test_loader_plan_uses_vertex_id_index(self):
        from evaluation.core import LOAD_EDGES_QUERY
        with self.db.driver.session() as s:
            plan = s.run('EXPLAIN '+LOAD_EDGES_QUERY, gid=self.gid,
                         es=[dict(a=0, b=1, t=1)]).data()
        scans = [str(r) for r in plan if 'Scan' in str(r)]
        self.assertEqual(len(scans), 2, plan)
        self.assertTrue(all(':Node {id}' in r for r in scans), plan)


    def test_batched_pivot_neighbor_groups(self):
        for groups, existing in product(product((False, True), repeat=3), (False, True)):
            with self.subTest(groups=groups, existing=existing):
                g = pivot_neighborhood(groups, existing)
                load(self.db, g, self.gid)
                self.assertEqual(self.db.pivot_rule_batched(self.gid), 1)
                self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(self.db.pivot_rule_batched(self.gid), 0)

    def test_pivot_eligibility_markers_are_scoped_and_cleared(self):
        foreign = self.gid + '_foreign'
        other = build(dict(family='identity', size=1, seed=0))
        try:
            load(self.db, other, foreign)
            for active in (False, True):
                graph = pivot_neighborhood((True, True, True), True) if active else other
                load(self.db, graph, self.gid)
                # Simulate stale markers from an interrupted query, including
                # invalid candidates and an unrelated graph using the label.
                with self.db.driver.session() as s:
                    s.run('MATCH(n:Node) WHERE n.graph_id IN $gids '
                          'SET n:_ZXdbPivotEligible', gids=[self.gid, foreign]).consume()
                count = self.db.apply_rule_optimized('pivot_rule', self.gid, exclusive_access=True)
                self.assertEqual(count, int(active))
                self.assertEqual(verify(graph, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(payload(read(self.db, foreign)), payload(other))
                with self.db.driver.session() as s:
                    self.assertEqual(s.run('MATCH(n:_ZXdbPivotEligible {graph_id:$gid}) '
                                           'RETURN count(n) AS n', gid=self.gid).single()['n'], 0)
                    self.assertEqual(s.run('MATCH(n:_ZXdbPivotEligible {graph_id:$gid}) '
                                           'RETURN count(n) AS n', gid=foreign).single()['n'], other.num_vertices())
        finally:
            self.db.empty_graphdb(foreign)

    def test_pauli_boundary_eligibility_markers_are_scoped_and_cleared(self):
        from tests.test_boundary_pivot_batch import boundary_neighborhood
        foreign = self.gid + '_foreign'
        other = build(dict(family='identity', size=1, seed=0))
        try:
            load(self.db, other, foreign)
            for active in (False, True):
                graph = boundary_neighborhood((True, True, True), True, a_phase=1) if active else other
                load(self.db, graph, self.gid)
                # Simulate stale markers from an interrupted query, including
                # invalid candidates and an unrelated graph using the label.
                with self.db.driver.session() as s:
                    s.run('MATCH(n:Node) WHERE n.graph_id IN $gids '
                          'SET n:_ZXdbPivotEligible', gids=[self.gid, foreign]).consume()
                with self.db.driver.session() as session:
                    count = self.db._run_count(session, self.db._query('Pivot boundary Pauli - batched'),
                                               'interior_pauli_removed', self.gid)
                self.assertEqual(count, int(active))
                self.assertEqual(verify(graph, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(payload(read(self.db, foreign)), payload(other))
                with self.db.driver.session() as s:
                    self.assertEqual(s.run('MATCH(n:_ZXdbPivotEligible {graph_id:$gid}) '
                                           'RETURN count(n) AS n', gid=self.gid).single()['n'], 0)
                    self.assertEqual(s.run('MATCH(n:_ZXdbPivotEligible {graph_id:$gid}) '
                                           'RETURN count(n) AS n', gid=foreign).single()['n'], other.num_vertices())
        finally:
            self.db.empty_graphdb(foreign)

    def test_grouped_supplementarity_matches_control_counts_and_semantics(self):
        from evaluation.cases import RULES
        cases = [(name, builder()) for name, builder, *_ in RULES['supplementarity']['cases']]
        cases += [(f'copies_{n}', build(dict(family='copies_supplementarity', size=n, seed=0)))
                  for n in (2, 3)]
        for name, graph in cases:
            with self.subTest(case=name):
                load(self.db, graph, self.gid)
                expected = self.db.supplementarity_simp(self.gid)
                load(self.db, graph, self.gid)
                actual = self.db.apply_rule_optimized('supplementarity_simp', self.gid,
                                                      exclusive_access=True)
                self.assertEqual(actual, expected)
                self.assertEqual(verify(graph, read(self.db, self.gid))['status'], 'verified')

    def test_batched_toggles_preserve_simple_neighbor_edges(self):
        # All absent/simple/H combinations between three neighbors. A simple
        # edge must survive adding H; deleting it changes the represented map.
        for method in ('pivot_rule', 'pivot_gadget_rule', 'local_complementation_rule',
                       'pivot_boundary_rule', 'pauli_boundary'):
            for kinds in product((0, 1, 2), repeat=3):
                with self.subTest(method=method, kinds=kinds):
                    if method == 'local_complementation_rule':
                        g = zx.Graph()
                        center = g.add_vertex(zx.VertexType.Z, phase=Fraction(1, 2))
                        neighbors, outputs = [], []
                        for _ in range(3):
                            n = g.add_vertex(zx.VertexType.Z, phase=Fraction(1, 4))
                            b = g.add_vertex(zx.VertexType.BOUNDARY)
                            g.add_edge((center, n), zx.EdgeType.HADAMARD)
                            g.add_edge((n, b))
                            neighbors.append(n); outputs.append(b)
                        g.set_outputs(tuple(outputs))
                    else:
                        g = pivot_neighborhood((True, True, True), False)
                        neighbors = [v for v in g.vertices() if v > 1 and g.type(v) == zx.VertexType.Z]
                        if method in ('pivot_gadget_rule', 'pivot_boundary_rule'):
                            g.set_phase(1, Fraction(3, 4))
                        if method in ('pivot_boundary_rule', 'pauli_boundary'):
                            b = g.add_vertex(zx.VertexType.BOUNDARY)
                            g.add_edge((1, b))
                            g.set_outputs(g.outputs() + (b,))
                    for pair, kind in zip(combinations(neighbors, 2), kinds):
                        if kind:
                            g.add_edge(pair, zx.EdgeType(kind))
                    load(self.db, g, self.gid)
                    rule = 'pivot_rule' if method == 'pauli_boundary' else method
                    self.assertGreater(self.db.apply_rule_optimized(rule, self.gid, exclusive_access=True), 0)
                    self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')

    def test_batched_pivot_disjoint_calls_and_reference(self):
        g = build(dict(family='copies_pivot', size=32, seed=0))
        load(self.db, g, self.gid)
        counter = CountingDriver(self.db.driver)
        self.db._driver = counter
        try:
            self.assertEqual(self.db.pivot_rule_batched(self.gid), 32)
            self.assertEqual(counter.submissions, 4)  # ownership + batch + empty + boundary
        finally:
            self.db._driver = counter.driver
        result = read(self.db, self.gid)
        reference(g, 'pivot', time.monotonic()+30)
        self.assertEqual(compare_outputs(result, g, max_iso_vertices=1024, iso_timeout_s=10)['graph']['status'], 'equal')

    def test_batched_full_reduce_overlapping_circuits(self):
        for family in ('random_clifford', 'random_clifford_t', 'random_t_heavy'):
            for seed in range(6):
                with self.subTest(family=family, seed=seed):
                    original = build(dict(family=family, size=2, depth=6, seed=seed))
                    load(self.db, original, self.gid)
                    self.db.full_reduce_batched_pivot(self.gid)
                    result = read(self.db, self.gid)
                    self.assertNotEqual(verify(original, result)['status'], 'failed')
                    expected = original.copy(); zx.full_reduce(expected)
                    self.assertEqual(verify(expected, result)['status'], 'verified')

    def test_batched_lcomp_disjoint_and_overlapping(self):
        from evaluation.cases import lc_two_candidates_shared
        for graph in (build(dict(family='lcomp', size=2, seed=0)), lc_two_candidates_shared()):
            load(self.db, graph, self.gid)
            self.db.local_complementation_rule_batched(self.gid)
            self.assertEqual(verify(graph, read(self.db, self.gid))['status'], 'verified')
        graph = build(dict(family='copies_lcomp', size=32, seed=0))
        load(self.db, graph, self.gid)
        counter = CountingDriver(self.db.driver)
        self.db._driver = counter
        try:
            self.assertEqual(self.db.local_complementation_rule_batched(self.gid), 32)
            self.assertEqual(counter.submissions, 3)
        finally:
            self.db._driver = counter.driver
        result = read(self.db, self.gid)
        reference(graph, 'lcomp', time.monotonic()+30)
        self.assertEqual(compare_outputs(result, graph, max_iso_vertices=1024, iso_timeout_s=10)['graph']['status'], 'equal')


if __name__ == '__main__':
    unittest.main()
