"""Optional live audit for the retained Python Memgraph query module."""
import os
import unittest

from evaluation.cases import RULES, pg_connected_neighbors
from evaluation.core import load, read
from evaluation.workloads import payload
from tests.live_support import ResearchLiveTests
from tests.test_rule_audit import adversarial_context, assert_projective


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1' and
                     os.environ.get('ZXDB_PYTHON_MODULE') == '1',
                     'Requires deployed zxqm.py and ZXDB_PYTHON_MODULE=1')
class PythonGadgetTests(ResearchLiveTests):
    def setUp(self):
        super().setUp()
        with self.db.driver.session() as session:
            session.run('CALL mg.load_all()').consume()

    def apply(self):
        with self.db.driver.session() as session:
            return session.run('CALL zxqm.pivot_gadget_fixpoint($gid) '
                               'YIELD applied RETURN applied', gid=self.gid).single()['applied']

    def test_rule_contexts_and_simple_neighbor_edges(self):
        graphs = [(name, adversarial_context(builder()))
                  for name, builder, *_ in RULES['pivot_gadget']['cases']]
        g = pg_connected_neighbors()
        # Change only a wire between the pivot's external neighbors.
        centers = {v for v in g.vertices() if g.vertex_degree(v) > 1
                   and all(g.type(n) == 1 for n in g.neighbors(v))}
        for e in g.edges():
            a, b = g.edge_st(e)
            if a not in centers and b not in centers and g.type(a) == g.type(b) == 1:
                g.set_edge_type(e, 1)
        graphs.append(('simple_neighbor', g))
        for name, g in graphs:
            with self.subTest(case=name):
                load(self.db, g, self.gid)
                self.apply()
                assert_projective(self, g, read(self.db, self.gid))
                self.assertEqual(self.apply(), 0)

    def test_explicit_graph_id_preserves_foreign_graph(self):
        foreign = self.gid + '_foreign'
        try:
            g = pg_connected_neighbors()
            load(self.db, g, self.gid)
            load(self.db, g, foreign)
            before = payload(read(self.db, foreign))
            self.assertGreater(self.apply(), 0)
            self.assertEqual(payload(read(self.db, foreign)), before)
            assert_projective(self, g, read(self.db, self.gid))
        finally:
            self.db.empty_graphdb(foreign)

    def test_cross_graph_edges_are_rejected_before_mutation(self):
        foreign = self.gid + '_foreign'
        try:
            load(self.db, pg_connected_neighbors(), self.gid)
            load(self.db, pg_connected_neighbors(), foreign)
            before = payload(read(self.db, self.gid))
            other = payload(read(self.db, foreign))
            with self.db.driver.session() as session:
                session.run('MATCH(a:Node {graph_id:$gid}), (b:Node {graph_id:$other}) '
                            'WITH a,b LIMIT 1 CREATE(a)-[:Wire {t:2}]->(b)',
                            gid=self.gid, other=foreign).consume()
            with self.assertRaisesRegex(Exception, 'Cross-graph'):
                self.apply()
            self.assertEqual(payload(read(self.db, self.gid)), before)
            self.assertEqual(payload(read(self.db, foreign)), other)
        finally:
            self.db.empty_graphdb(foreign)
