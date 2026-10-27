"""Live Memgraph regressions for diagram ownership (no whole-DB clearing)."""
import os
import unittest
import uuid

from evaluation.cases import RULES, sf_chain3, sf_xx, pg_basic
from evaluation.circuits import random_circuit_graph
from evaluation.oracle import _quimb_equal
from pathlib import Path
from tempfile import TemporaryDirectory
from zxdb.zxdb import ZXdb


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1', 'Opt-in Memgraph checks')
class TestGraphIsolation(unittest.TestCase):
    def setUp(self):
        self.db = ZXdb(uri=os.environ.get('ZXDB_URI', 'bolt://localhost:7687'),
                       user=os.environ.get('ZXDB_USER', ''), password=os.environ.get('ZXDB_PASSWORD', ''))
        prefix = 'isolation_' + uuid.uuid4().hex
        self.target, self.other, self.missing = [prefix + s for s in ('_a', '_b', '_missing')]

    def tearDown(self):
        for gid in (self.target, self.other):
            self.db.empty_graphdb(gid)
        self.db.close()

    def load(self, g, gid):
        # Deliberately bypass the separately audited importer, which still
        # clears the whole database with initialize_empty=True.
        with self.db.driver.session() as s:
            s.run('UNWIND $vs AS v CREATE (n:Node) SET n = v, n.graph_id = $gid',
                  vs=[{'id': int(v), 't': int(g.type(v)), 'phase': float(g.phase(v))}
                      for v in g.vertices()], gid=gid).consume()
            s.run('UNWIND $es AS e MATCH (a:Node {graph_id:$gid,id:e.a}), '
                  '(b:Node {graph_id:$gid,id:e.b}) '
                  'CREATE (a)-[:Wire {t:e.t,graph_id:$gid}]->(b)',
                  es=[{'a': int(g.edge_s(e)), 'b': int(g.edge_t(e)),
                       't': int(g.edge_type(e))} for e in g.edges()], gid=gid).consume()
            for label, vertices in [('Input', g.inputs()), ('Output', g.outputs())]:
                s.run('UNWIND $vs AS v MATCH (n:Node {graph_id:$gid,id:v.id}) '
                      f'SET n:{label}, n.io_index=v.idx',
                      vs=[{'id': int(v), 'idx': i} for i, v in enumerate(vertices)],
                      gid=gid).consume()

    def snapshot(self, gid):
        # Check stored IDs, labels, all properties and incident relationships;
        # export/isomorphism would hide changes to markers, edge types or IDs.
        with self.db.driver.session() as s:
            nodes = s.run('MATCH (n {graph_id:$gid}) RETURN id(n) AS id, '
                          'labels(n) AS labels, properties(n) AS props ORDER BY id',
                          gid=gid).data()
            for n in nodes:
                n['labels'].sort()
            edges = s.run('MATCH (a)-[r]->(b) '
                          'WHERE a.graph_id=$gid OR b.graph_id=$gid '
                          'RETURN id(r) AS id, id(a) AS a, id(b) AS b, '
                          'type(r) AS type, properties(r) AS props ORDER BY id',
                          gid=gid).data()
            return nodes, edges

    def test_every_rule_preserves_second_diagram(self):
        for name, spec in RULES.items():
            fired = False
            for case in spec['cases']:
                with self.subTest(rule=name, case=case[0]):
                    self.load(case[1](), self.target)
                    self.load(case[1](), self.other)
                    before = self.snapshot(self.other)
                    count = getattr(self.db, spec['db_method'])(self.target)
                    fired |= bool(count)
                    self.assertEqual(self.snapshot(self.other), before)
                    self.db.empty_graphdb(self.target)
                    self.db.empty_graphdb(self.other)
            self.assertTrue(fired, f'{name} never fired in the isolation fixtures')

    def test_composed_and_auxiliary_rules(self):
        for method, builder in [
            ('full_reduce', sf_chain3), ('full_reduce', pg_basic),
            ('full_reduce_batched_pivot', pg_basic),
            ('interior_clifford_simp', sf_chain3),
            ('clifford_simp', sf_chain3), ('to_gh', sf_xx),
            ('pivot_rule_batched', RULES['pivot']['cases'][0][1]),
        ]:
            with self.subTest(method=method, builder=builder.__name__):
                self.load(builder(), self.target)
                self.load(builder(), self.other)
                before = self.snapshot(self.other)
                target_before = self.snapshot(self.target)
                getattr(self.db, method)(self.target)
                self.assertEqual(self.snapshot(self.other), before)
                self.assertNotEqual(self.snapshot(self.target), target_before)
                self.db.empty_graphdb(self.target)
                self.db.empty_graphdb(self.other)

    def test_missing_graph_is_noop_for_all_query_entries(self):
        self.load(sf_chain3(), self.other)
        with self.db.driver.session() as s:
            # Shared marker values must not cause cross-graph cleanup or joins.
            s.run('MATCH (n:Node {graph_id:$gid}) '
                  'SET n.pattern_id="foreign", n.bialg_side=n.id % 2',
                  gid=self.other).consume()
            before = self.snapshot(self.other)
            for title in self.db.basic_rewrite_rule_queries:
                with self.subTest(query=title):
                    s.run(self.db._query(title), graph_id=self.missing).consume()
                    self.assertEqual(self.snapshot(self.other), before)
        self.db.full_reduce(self.missing)
        self.assertEqual(self.snapshot(self.other), before)

    def test_bialgebra_connect_keeps_foreign_markers(self):
        with self.db.driver.session() as s:
            for gid in (self.target, self.other):
                s.run('CREATE (:Node {graph_id:$gid,t:1,phase:0,bialg_side:0}), '
                      '(:Node {graph_id:$gid,t:2,phase:0,bialg_side:1})', gid=gid).consume()
            before = self.snapshot(self.other)
            s.run(self.db._query('Bialgebra connect'), graph_id=self.target).consume()
            self.assertEqual(self.snapshot(self.other), before)
            nodes, edges = self.snapshot(self.target)
            self.assertEqual(len(edges), 1)
            self.assertTrue(all('bialg_side' not in n['props'] for n in nodes))

    def test_isolated_cleanup_is_scoped(self):
        with self.db.driver.session() as s:
            for gid in (self.target, self.other):
                s.run('CREATE (:Node {graph_id:$gid,t:1,phase:0})', gid=gid).consume()
        before = self.snapshot(self.other)
        self.assertEqual(self.db.remove_isolated_vertices(self.target), 1)
        self.assertEqual(self.snapshot(self.target), ([], []))
        self.assertEqual(self.snapshot(self.other), before)

    def test_cross_graph_relationship_is_rejected_before_mutation(self):
        self.load(sf_chain3(), self.target)
        self.load(sf_chain3(), self.other)
        with self.db.driver.session() as s:
            s.run('MATCH (a:Node {graph_id:$a,id:1}), (b:Node {graph_id:$b,id:1}) '
                  'CREATE (a)-[:Wire {t:1}]->(b)', a=self.target, b=self.other).consume()
        before = self.snapshot(self.target), self.snapshot(self.other)
        for method in ('full_reduce', 'remove_identities', 'spider_fusion',
                       'to_gh', 'hadamard_cancel'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'outside that graph'):
                    getattr(self.db, method)(self.target)
                self.assertEqual((self.snapshot(self.target), self.snapshot(self.other)), before)

    def test_hadamard_session_helper_and_wire_ownership(self):
        builder = RULES['hadamard_cancel']['cases'][0][1]
        self.load(builder(), self.target)
        self.load(builder(), self.other)
        before = self.snapshot(self.other)
        with self.db.driver.session() as s:
            self.assertGreater(self.db.hadamard_cancel_fn(self.target, s), 0)
        self.assertEqual(self.snapshot(self.other), before)
        self.assertTrue(all(e['props'].get('graph_id') == self.target
                            for e in self.snapshot(self.target)[1]))

    def test_native_wrapper_refuses_mixed_graphs_before_any_mutation(self):
        self.load(sf_chain3(), self.target)
        self.load(pg_basic(), self.other)
        before = self.snapshot(self.target), self.snapshot(self.other)
        for method in ('pivot_gadget_rule_qm', 'full_reduce_with_query_modules'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'only the requested graph'):
                    getattr(self.db, method)(self.target)
                getattr(self.db, method)(self.missing)
                self.assertEqual((self.snapshot(self.target), self.snapshot(self.other)), before)

    def test_boundary_pivot_composition_preserves_tensor_with_scoped_matches(self):
        # Filtering already-bound nodes inside pivot neighborhood MATCHes can
        # change the analytical-mode plan and corrupt boundary connectivity.
        # Exercise both graph insertion orders and several database ID layouts.
        with TemporaryDirectory() as directory:
            for iteration in range(6):
                with self.subTest(iteration=iteration):
                    g = random_circuit_graph(1005, 2, 6)
                    gids = (self.target, self.other) if iteration % 2 else (self.other, self.target)
                    for gid in gids:
                        self.load(g, gid)
                    before = self.snapshot(self.other)
                    self.db.full_reduce(self.target)
                    reduced = self.db.export_graphdb_to_zx_graph(
                        self.target, str(Path(directory) / 'reduced.json'))
                    self.assertIs(_quimb_equal(g, reduced), True)
                    self.assertEqual(self.snapshot(self.other), before)
                    self.db.empty_graphdb(self.target)
                    self.db.empty_graphdb(self.other)


if __name__ == '__main__':
    unittest.main()
