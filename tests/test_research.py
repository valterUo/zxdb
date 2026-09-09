"""Fast suite contracts; live checks opt in with ZXDB_RESEARCH_LIVE=1."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import uuid
from types import SimpleNamespace

import pyzx as zx

from evaluation.research_workloads import (FAMILIES, RULES, build, digest, make_plan,
                                           metrics, opportunities, payload, rule_for)
from evaluation.research_core import (CountingDriver, load, read, reference,
                                      validate_graph, verify, compare_outputs)
from evaluation.research_report import interval, report, summarize


class ResearchTests(unittest.TestCase):
    def test_database_variant_dispatch_and_explicit_exclusive_contract(self):
        from evaluation.research import reduce_database
        db = Mock()
        reduce_database(db, 'full_reduce', 'g', 'optimized')
        db.full_reduce_optimized.assert_called_once_with('g', exclusive_access=True, incremental=False)
        reduce_database(db, 'pivot', 'g', 'worklist')
        db.apply_rule_optimized.assert_called_once_with('pivot_rule', 'g', exclusive_access=True, incremental=True)
        reduce_database(db, 'pivot', 'g', 'control')
        db.pivot_rule.assert_called_once_with(graph_id='g')
        with self.assertRaises(ValueError):
            reduce_database(db, 'pivot', 'g', 'unknown')
    def test_native_baseline_has_no_fingerprint_or_redundant_pass(self):
        for rule in RULES:
            batch = rule in ('pivot_gadget', 'pivot_boundary', 'gadget_fusion')
            fn = Mock(side_effect=[True, True, False] if batch else [True])
            with self.subTest(rule=rule), patch.object(zx.simplify, RULES[rule][1], fn), \
                    patch('evaluation.research_core.payload', side_effect=AssertionError('Timed serialization')):
                reference(zx.Graph(), rule, time.monotonic()+10)
                self.assertEqual(fn.call_count, 3 if batch else 1)

    def test_projective_oracle_handles_tiny_nonzero_and_actual_zero_scalars(self):
        original = build(dict(family='identity', size=1, seed=0))
        tiny = original.copy(); tiny.scalar.power2 = -10000
        self.assertEqual(verify(original, tiny)['status'], 'verified')
        tiny.scalar.is_zero = True
        self.assertEqual(verify(original, tiny)['status'], 'failed')
        unknown = original.copy(); unknown.scalar.is_unknown = True
        self.assertEqual(verify(original, unknown)['status'], 'inconclusive')

    def test_linear_copies_match_tensor_products(self):
        for rule in RULES:
            if rule == 'full_reduce':
                continue
            motif = build(dict(family=rule, size=2, seed=0))
            expected = motif.tensor(motif).tensor(motif)
            actual = build(dict(family='copies_'+rule, size=3, seed=0))
            self.assertEqual(payload(actual), payload(expected), rule)
        # Construction must not call tensor(), whose prefix copies are quadratic.
        with patch.object(type(zx.Graph()), 'tensor', side_effect=AssertionError('Quadratic construction')):
            large = build(dict(family='copies_spider_fusion', size=8192, seed=0))
        self.assertEqual(large.num_vertices(), 65536)
        plan = make_plan('massive')
        self.assertTrue(any(s['size'] == 131072 for s in plan['cases']))
        self.assertTrue(any(s['depth'] == 32768 for s in plan['cases']))

    def test_isolated_pyzx_success_timeout_and_memory_limit(self):
        from evaluation.research_pyzx import isolated_reference
        original = build(dict(family='identity', size=1, seed=0))
        result, info = isolated_reference(original, 'identity', 30, 1024)
        self.assertEqual(info['status'], 'ok')
        self.assertEqual(verify(original, result)['status'], 'verified')
        for seconds, memory, status in [(.001, 1024, 'timeout'), (30, 1, 'memory_limit')]:
            result, info = isolated_reference(original, 'identity', seconds, memory)
            self.assertIsNone(result)
            self.assertEqual(info['status'], status)
            self.assertIsNone(info['reduce_s'])

    def test_capacity_results_do_not_invent_speedups(self):
        plan = make_plan(); plan['cases'] = plan['cases'][:1]
        rows = [dict(kind='trial', case=plan['cases'][0], trial=i, warmup=False,
                     status='resource_limit', engines={
                         'db': dict(reduce_s=2, verification=dict(status='inconclusive')),
                         'pyzx': dict(reduce_s=None, status='timeout', verification=dict(status='inconclusive'))})
                for i in range(2)]
        result = summarize(plan, rows)[0]
        self.assertEqual(result['db_completed_median_s'], 2)
        self.assertEqual(result['pyzx_timeout_trials'], 2)
        self.assertIsNone(result['speedup'])
        self.assertFalse(result['complete'])

    def test_output_comparison_levels_and_boundary_order(self):
        original = build(dict(family='identity', size=2, seed=0))
        same = compare_outputs(original, original.copy())
        self.assertEqual(same['tensor']['status'], 'verified')
        self.assertEqual(same['graph']['status'], 'equal')
        changed = original.copy(); changed.set_phase(1, 1)
        comparison = compare_outputs(original, changed)
        self.assertEqual(comparison['tensor']['status'], 'failed')
        self.assertEqual(comparison['graph']['status'], 'different')
        self.assertEqual(comparison['degree_sequence']['status'], 'equal')
        swapped = original.copy(); swapped.set_outputs(tuple(reversed(swapped.outputs())))
        self.assertEqual(compare_outputs(original, swapped)['graph']['status'], 'different')

    def test_output_isomorphism_retains_parallel_wire_type_multiplicity(self):
        def graph(types):
            g = zx.Graph(backend='multigraph'); g.set_auto_simplify(False)
            a = g.add_vertex(zx.VertexType.Z, phase=.25)
            b = g.add_vertex(zx.VertexType.X, phase=.25)
            for t in types:
                g.add_edge((a, b), zx.EdgeType(t))
            return g
        result = compare_outputs(graph([1, 1, 2]), graph([1, 2, 2]))
        self.assertEqual(result['degree_sequence']['status'], 'equal')
        self.assertEqual(result['graph']['status'], 'different')

    def test_structural_disagreement_is_not_semantic_failure(self):
        original = build(dict(family='identity', size=2, seed=0))
        reduced = original.copy(); reference(reduced, 'identity', time.monotonic()+10)
        comparison = compare_outputs(original, reduced)
        self.assertEqual(comparison['tensor']['status'], 'verified')
        self.assertEqual(comparison['graph']['status'], 'different')
        self.assertEqual(comparison['degree_sequence']['status'], 'different')

    def test_structural_limits_leave_cheap_degree_evidence_available(self):
        original = build(dict(family='identity', size=100, seed=0))
        result = compare_outputs(original, original)
        self.assertEqual(result['tensor']['status'], 'inconclusive')
        self.assertEqual(result['graph']['status'], 'inconclusive')
        self.assertEqual(result['degree_sequence']['status'], 'equal')
        small = build(dict(family='identity', size=1, seed=0))
        self.assertEqual(compare_outputs(small, small, iso_timeout_s=-1)['graph']['status'], 'inconclusive')

    def test_fixed_plan_and_reproducible_inputs(self):
        self.assertEqual(make_plan(), make_plan())
        for family in FAMILIES:
            spec = dict(family=family, size=2, depth=2, seed=3)
            with self.subTest(family=family):
                a, b = build(spec), build(spec)
                validate_graph(a)
                self.assertEqual(digest(payload(a)), digest(payload(b)))

    def test_every_rule_family_actually_applies_at_multiple_scales(self):
        for family in FAMILIES:
            if rule_for(family) == 'full_reduce':
                continue
            for size in (2, 8):
                with self.subTest(family=family, size=size):
                    spec = dict(family=family, size=size, depth=None, seed=0)
                    graph = build(spec); before = payload(graph)
                    self.assertGreater(opportunities(spec), 0)
                    reference(graph, rule_for(family), time.monotonic()+10)
                    self.assertNotEqual(before, payload(graph))

    def test_small_rule_motifs_preserve_semantics_in_reference(self):
        for family in RULES:
            if family == 'full_reduce':
                continue
            with self.subTest(family=family):
                original = build(dict(family=family, size=2, depth=None, seed=0))
                result = original.copy()
                reference(result, family, time.monotonic()+10)
                self.assertEqual(verify(original, result)['status'], 'verified')

    def test_realistic_full_reduce_reference(self):
        for family in ('random_clifford', 'random_clifford_t', 'random_t_heavy'):
            for seed in range(4):
                with self.subTest(family=family, seed=seed):
                    original = build(dict(family=family, size=2, depth=3, seed=seed))
                    result = original.copy()
                    reference(result, 'full_reduce', time.monotonic()+10)
                    self.assertEqual(verify(original, result)['status'], 'verified')

    def test_locality_preserves_edge_budget_and_rejects_impossible_request(self):
        spec = dict(family='layered_local', size=5, depth=3, seed=42, edges_per_layer=4)
        local, global_ = build(spec), build(dict(spec, family='layered_global'))
        self.assertEqual(metrics(local)['edges'], metrics(global_)['edges'])
        with self.assertRaises(ValueError):
            build(dict(spec, edges_per_layer=5))

    def test_topology_fingerprint_detects_rewiring(self):
        g = build(dict(family='identity', size=2, seed=0))
        h = g.copy(); edges = list(h.edges())
        a, b = h.edge_st(edges[0]); c, d = h.edge_st(edges[-1])
        h.remove_edges([edges[0], edges[-1]]); h.add_edges([(a, d), (c, b)])
        self.assertNotEqual(payload(g), payload(h))

    def test_oracle_wrong_phase_zero_and_resource_limit(self):
        a = build(dict(family='identity', size=1, seed=0))
        b = a.copy(); b.set_phase(1, 1)
        self.assertEqual(verify(a, b)['status'], 'failed')
        zero = a.copy(); zero.add_vertex(zx.VertexType.Z, phase=1)
        self.assertEqual(verify(zero, a)['status'], 'failed')
        self.assertEqual(verify(zero, zero)['status'], 'verified')
        large = build(dict(family='identity', size=100, seed=0))
        self.assertEqual(verify(large, large)['status'], 'inconclusive')

    def test_counter_covers_autocommit_and_managed_calls(self):
        driver, session, tx = Mock(), Mock(), Mock()
        driver.session.return_value = session
        session.execute_write.side_effect = lambda fn: fn(tx)
        counter = CountingDriver(driver)
        proxy = counter.session()
        proxy.run('RETURN 1')
        proxy.execute_write(lambda t: t.run('RETURN 2'))
        self.assertEqual(counter.submissions, 2)
        self.assertEqual(counter.last_query, 'RETURN 2')
        counter.deadline = 0
        with self.assertRaises(TimeoutError):
            proxy.run('RETURN 3')

    def test_paired_estimator_direction_and_cluster_weighting(self):
        self.assertEqual(interval([[2, 2], [8, 8]])[0], 4)
        self.assertEqual(interval([[2]*100, [8]*2])[0], 4)
        self.assertEqual(interval([[2, 2]])[:], (2, 2, 2))

    def test_reporting_excludes_warmups_and_does_not_hide_failed_trials(self):
        plan = make_plan(); plan['cases'] = plan['cases'][:1]
        spec = plan['cases'][0]
        def trial(index, status='ok'):
            return dict(kind='trial', case=spec, trial=index, warmup=index < 0, status=status,
                        engines={e: dict(reduce_s=t, verification=dict(status='verified'))
                                 for e, t in [('db', 2), ('pyzx', 4)]})
        rows = [trial(-1), trial(0), trial(1)]
        self.assertEqual(summarize(plan, rows)[0]['speedup'], 2)
        rows[0]['status'] = 'failed'
        self.assertIsNone(summarize(plan, rows)[0]['speedup'])
        rows[0]['status'] = 'ok'
        rows[-1]['status'] = 'no_progress'
        self.assertIsNone(summarize(plan, rows)[0]['speedup'])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); (p / 'blocks').mkdir()
            (p / 'plan.json').write_text(json.dumps(plan))
            (p / 'blocks' / 'test.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            report(p)
            self.assertIn('nan', next((p / 'figures').glob('*.dat')).read_text())
            self.assertIn('pgfplots', (p / 'figures' / 'figures.tex').read_text())


@unittest.skipUnless(os.environ.get('ZXDB_RESEARCH_LIVE') == '1', 'Opt-in dedicated Memgraph validation')
class ResearchLiveTests(unittest.TestCase):
    def setUp(self):
        from neo4j import GraphDatabase
        from zxdb.zxdb import ZXdb
        uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
        auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as s:
                if s.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                    self.skipTest('Use an empty dedicated database')
        self.db = ZXdb(uri=uri, user=auth[0], password=auth[1])
        self.gid = 'research_test_' + uuid.uuid4().hex

    def tearDown(self):
        self.db.empty_graphdb(self.gid); self.db.close()

    def test_rule_motifs_and_realistic_full_reduce(self):
        cases = [(r, r, 2, None, 0) for r in RULES if r != 'full_reduce']
        cases += [(f, 'full_reduce', 2, 3, seed) for f in
                  ('random_clifford', 'random_clifford_t', 'random_t_heavy') for seed in range(4)]
        for family, rule, size, depth, seed in cases:
            with self.subTest(family=family, seed=seed):
                original = build(dict(family=family, size=size, depth=depth, seed=seed))
                load(self.db, original, self.gid)
                getattr(self.db, RULES[rule][0])(graph_id=self.gid)
                result = read(self.db, self.gid)
                self.assertEqual(verify(original, result)['status'], 'verified')
                if rule != 'full_reduce':
                    self.assertNotEqual(metrics(original), metrics(result))

    def test_loader_reader_preserve_parallel_edges_and_boundary_order(self):
        original = zx.Graph(backend='multigraph'); original.set_auto_simplify(False)
        bi = original.add_vertex(zx.VertexType.BOUNDARY)
        a = original.add_vertex(zx.VertexType.Z, phase=0.25)
        b = original.add_vertex(zx.VertexType.X, phase=0.25)
        bo = original.add_vertex(zx.VertexType.BOUNDARY)
        original.add_edges([(bi, a), (a, b), (b, bo)])
        original.add_edge((a, b), zx.EdgeType.HADAMARD)
        original.add_edge((a, a), zx.EdgeType.HADAMARD)
        original.set_inputs((bi,)); original.set_outputs((bo,))
        load(self.db, original, self.gid)
        result = read(self.db, self.gid)
        self.assertEqual(metrics(original), metrics(result))
        self.assertEqual(verify(original, result)['status'], 'verified')

    def test_worker_report_pipeline_on_two_tiny_inputs(self):
        from evaluation.research import run
        plan = make_plan()
        plan['db_variant'] = 'optimized'
        plan['cases'] = [plan['cases'][0], next(s for s in plan['cases'] if s['family'] == 'random_clifford_t')]
        plan['cases'][1]['depth'] = 3
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); plan_path = p / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            output = p / 'run'
            run(SimpleNamespace(output=str(output), plan=str(plan_path), budget_seconds=360))
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(r['complete'] and r['evidence'] == 'verified' for r in summary), summary)
            self.assertTrue(all(r['outputs_tensor_verified'] == plan['repetitions'] for r in summary))
            self.assertTrue(all(r['outputs_degree_sequence_equal'] + r['outputs_degree_sequence_different']
                                == plan['repetitions'] for r in summary))

    def test_pyzx_timeout_keeps_database_trials_and_completes_block(self):
        from evaluation.research import run
        plan = make_plan()
        plan['cases'] = plan['cases'][:1]
        plan.update(pyzx_timeout_s=.001, pyzx_max_rss_mb=1024, applicability_probe='motif')
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); plan_path = p / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            output = p / 'run'
            run(SimpleNamespace(output=str(output), plan=str(plan_path), budget_seconds=360))
            summary = json.loads((output / 'summary.json').read_text())[0]
            self.assertEqual(summary['db_completed_trials'], 2)
            self.assertEqual(summary['pyzx_timeout_trials'], 2)
            self.assertIsNone(summary['speedup'])
            rows = [json.loads(line) for line in next((output / 'blocks').glob('*.jsonl')).read_text().splitlines()]
            self.assertEqual(rows[-1]['kind'], 'complete')
            self.assertTrue(next((output / 'figures').glob('*-capacity.tikz')).exists())


if __name__ == '__main__':
    unittest.main()
