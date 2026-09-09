"""Optimized reduction regressions; live tests require an empty dedicated server."""
import os
import time
import unittest
from unittest.mock import patch, Mock
from itertools import product

import pyzx as zx

from evaluation.research_core import load, read, verify, reference, compare_outputs, CountingDriver
from evaluation.research_workloads import build, metrics
from tests.test_pivot_materialization import pivot_neighborhood
from tests.test_research import ResearchLiveTests


class WorklistUnitTests(unittest.TestCase):
    def test_cleanup_failure_does_not_hide_primary_error(self):
        from zxdb.worklists import fixpoint
        first = Mock()
        seed = Mock(); seed.single.return_value = {'n': 1}
        session = Mock(); session.run.side_effect = [first, seed, RuntimeError('Cleanup failed')]
        with patch('zxdb.worklists.query_for', side_effect=ValueError('Primary failure')):
            with self.assertRaisesRegex(ValueError, 'Primary failure'):
                fixpoint(session, '', 'lcomp', 'patterns_processed', 'g')


@unittest.skipUnless(os.environ.get('ZXDB_RESEARCH_LIVE') == '1', 'Opt-in Memgraph checks')
class ReductionOptimizationTests(unittest.TestCase):
    setUp = ResearchLiveTests.setUp
    tearDown = ResearchLiveTests.tearDown

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

    def test_worklists_revisit_changed_neighbors_and_clean_markers(self):
        from evaluation import cases
        builders = [('pivot', cases.pv_shared_neighbor), ('lcomp', cases.lc_two_candidates_shared)]
        for kind, builder in builders:
            graph = builder()
            load(self.db, graph, self.gid)
            method = self.db.pivot_rule_batched if kind == 'pivot' else self.db.local_complementation_rule_batched
            method(self.gid, exclusive_access=True, incremental=True)
            result = read(self.db, self.gid)
            self.assertEqual(verify(graph, result)['status'], 'verified')
            # The ordinary matcher must find no overlooked candidates.
            self.assertEqual(method(self.gid), 0)
            with self.db.driver.session() as s:
                self.assertEqual(s.run('MATCH(n) WHERE n:_ZXdbPivotWork OR n:_ZXdbLcompWork '
                                       'RETURN count(n) AS n').single()['n'], 0)

    def test_worklist_cleanup_after_injected_failure_and_stale_marker(self):
        from zxdb.worklists import fixpoint
        graph = build(dict(family='lcomp', size=2, seed=0))
        load(self.db, graph, self.gid)
        with self.db.driver.session() as s:
            s.run('MATCH(n:Node {graph_id:$gid}) SET n:_ZXdbLcompWork', gid=self.gid).consume()
            with patch('zxdb.worklists.query_for', side_effect=RuntimeError('Injected failure')):
                with self.assertRaises(RuntimeError):
                    fixpoint(s, '', 'lcomp', 'patterns_processed', self.gid)
            self.assertEqual(s.run('MATCH(n:_ZXdbLcompWork) RETURN count(n) AS n').single()['n'], 0)
        self.assertEqual(self.db.local_complementation_rule_batched(self.gid,
                         exclusive_access=True, incremental=True), 1)

    def test_worklists_find_candidates_created_by_prior_rewrite(self):
        g = zx.Graph()
        c = g.add_vertex(zx.VertexType.Z, phase=.5)
        v = g.add_vertex(zx.VertexType.Z)
        n = g.add_vertex(zx.VertexType.Z, phase=.25)
        out = g.add_vertex(zx.VertexType.BOUNDARY)
        g.add_edge((c, v), zx.EdgeType.HADAMARD)
        g.add_edge((v, n), zx.EdgeType.HADAMARD)
        g.add_edge((n, out)); g.set_outputs((out,))
        load(self.db, g, self.gid)
        self.assertEqual(self.db.local_complementation_rule_batched(self.gid,
                         exclusive_access=True, incremental=True), 2)
        self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')

    def test_worklist_preserves_foreign_graph_and_foreign_markers(self):
        from evaluation.research_workloads import payload
        graph = build(dict(family='lcomp', size=2, seed=0))
        foreign = self.gid + '_foreign'
        load(self.db, graph, self.gid); load(self.db, graph, foreign)
        try:
            with self.db.driver.session() as s:
                s.run('MATCH(n:Node {graph_id:$gid}) SET n:_ZXdbLcompWork', gid=foreign).consume()
            before = payload(read(self.db, foreign))
            self.db.full_reduce_optimized(self.gid, exclusive_access=True, incremental=True)
            self.assertEqual(payload(read(self.db, foreign)), before)
            with self.db.driver.session() as s:
                self.assertEqual(s.run('MATCH(n:_ZXdbLcompWork {graph_id:$gid}) RETURN count(n) AS n',
                                       gid=foreign).single()['n'], graph.num_vertices())
        finally:
            self.db.empty_graphdb(foreign)

    def test_repeated_boundary_load_preserves_graph_and_order(self):
        from evaluation.research_workloads import payload
        graph = build(dict(family='copies_lcomp', size=128, seed=0))
        expected = payload(graph)
        for attempt in range(80):
            load(self.db, graph, self.gid)
            self.assertEqual(payload(read(self.db, self.gid)), expected, attempt)

    def test_loader_plan_uses_vertex_id_index(self):
        from evaluation.research_core import LOAD_EDGES_QUERY
        with self.db.driver.session() as s:
            plan = s.run('EXPLAIN '+LOAD_EDGES_QUERY, gid=self.gid,
                         es=[dict(a=0, b=1, t=1)]).data()
        scans = [str(r) for r in plan if 'Scan' in str(r)]
        self.assertEqual(len(scans), 2, plan)
        self.assertTrue(all(':Node {id}' in r for r in scans), plan)

    def test_worklists_find_new_pivot_edge(self):
        g = zx.Graph()
        a, b, c, d = [g.add_vertex(zx.VertexType.Z) for _ in range(4)]
        for e in ((a, b), (a, c), (b, d)):
            g.add_edge(e, zx.EdgeType.HADAMARD)
        outs = []
        for v in (c, d):
            n = g.add_vertex(zx.VertexType.Z, phase=.25)
            out = g.add_vertex(zx.VertexType.BOUNDARY)
            g.add_edge((v, n), zx.EdgeType.HADAMARD)
            g.add_edge((n, out)); outs.append(out)
        g.set_outputs(tuple(outs))
        load(self.db, g, self.gid)
        self.assertEqual(self.db.pivot_rule_batched(self.gid,
                         exclusive_access=True, incremental=True), 2)
        self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')

    def test_batched_pivot_neighbor_groups(self):
        for groups, existing in product(product((False, True), repeat=3), (False, True)):
            with self.subTest(groups=groups, existing=existing):
                g = pivot_neighborhood(groups, existing)
                load(self.db, g, self.gid)
                self.assertEqual(self.db.pivot_rule_batched(self.gid), 1)
                self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(self.db.pivot_rule_batched(self.gid), 0)

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
