"""Typed parallel-edge algebra and the full_reduce audit counterexample."""
from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
import uuid

import numpy as np
import pyzx as zx

from evaluation.cases import sf_hadamard_parallel_via_fusion
from zxdb.zxdb import ZXdb


def dense_tensor(g):
    # Keep export's spring layout out of PyZX's contraction ordering. Preserve
    # all edges and graph scalars; do not use the audit's quimb tensor builder.
    h = g.copy()
    for i, v in enumerate(h.vertices()):
        h.set_row(v, 0 if v in h.inputs() else h.num_vertices() + 1
                  if v in h.outputs() else i + 1)
    return h.to_tensor()


def parallel_pair(colors, simple, hadamard):
    # Default GraphS eagerly simplifies parallel wires while building fixtures.
    # Retain the actual multiedges so the database, not PyZX, must normalize.
    g = zx.Graph(backend='multigraph')
    g.set_auto_simplify(False)
    bi = g.add_vertex(zx.VertexType.BOUNDARY)
    a = g.add_vertex(colors[0], phase=Fraction(1, 4))
    b = g.add_vertex(colors[1], phase=Fraction(1, 3))
    bo = g.add_vertex(zx.VertexType.BOUNDARY)
    g.add_edge((bi, a), zx.EdgeType.SIMPLE)
    for t, count in [(1, simple), (2, hadamard)]:
        for _ in range(count):
            g.add_edge((a, b), t)
    g.add_edge((b, bo), zx.EdgeType.SIMPLE)
    g.set_inputs((bi,))
    g.set_outputs((bo,))
    return g


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1', 'Opt-in Memgraph checks')
class TestHopfNormalization(unittest.TestCase):
    def setUp(self):
        self.db = ZXdb(uri=os.environ.get('ZXDB_URI', 'bolt://localhost:7687'),
                       user=os.environ.get('ZXDB_USER', ''), password=os.environ.get('ZXDB_PASSWORD', ''))
        self.gid = 'hopf_' + uuid.uuid4().hex
        self.foreign = self.gid + '_foreign'
        self.directory = TemporaryDirectory()

    def tearDown(self):
        self.db.empty_graphdb(self.gid)
        self.db.empty_graphdb(self.foreign)
        self.db.close()
        self.directory.cleanup()

    def load(self, g, gid=None):
        gid = self.gid if gid is None else gid
        self.db.empty_graphdb(gid)
        # The importer replacement bug is a separate audit item. Create only
        # our private fixture; alternate edge directions to check undirected ZX.
        with self.db.driver.session() as s:
            s.run('UNWIND $vs AS v CREATE (n:Node) SET n=v, n.graph_id=$gid',
                  vs=[{'id': int(v), 't': int(g.type(v)), 'phase': float(g.phase(v))}
                      for v in g.vertices()], gid=gid).consume()
            es = []
            for i, e in enumerate(g.edges()):
                a, b = g.edge_st(e)
                if i % 2:
                    a, b = b, a
                es.append({'a': int(a), 'b': int(b), 't': int(g.edge_type(e))})
            s.run('UNWIND $es AS e MATCH (a:Node {graph_id:$gid,id:e.a}), '
                  '(b:Node {graph_id:$gid,id:e.b}) '
                  'CREATE (a)-[:Wire {graph_id:$gid,t:e.t}]->(b)', es=es, gid=gid).consume()
            for label, vertices in [('Input', g.inputs()), ('Output', g.outputs())]:
                s.run('UNWIND $vs AS v MATCH(n:Node {graph_id:$gid,id:v.id}) '
                      f'SET n:{label}, n.io_index=v.i',
                      vs=[{'id': int(v), 'i': i} for i, v in enumerate(vertices)], gid=gid).consume()

    def export(self, gid=None):
        return self.db.export_graphdb_to_zx_graph(
            self.gid if gid is None else gid, str(Path(self.directory.name) / 'graph.json'))

    def assert_tensor_preserved(self, original):
        expected = dense_tensor(original).ravel()
        actual = dense_tensor(self.export()).ravel()
        self.assertEqual(actual.shape, expected.shape)
        if np.allclose(expected, 0, atol=1e-10):
            np.testing.assert_allclose(actual, 0, atol=1e-10)
        else:
            scale = np.vdot(expected, actual) / np.vdot(expected, expected)
            self.assertGreater(abs(scale), 1e-10)
            np.testing.assert_allclose(actual, scale * expected, atol=1e-9, rtol=1e-8)

    def internal_edge_types(self, gid=None):
        with self.db.driver.session() as s:
            return [r['t'] for r in s.run(
                'MATCH (a:Node {graph_id:$gid})-[r:Wire]->(b:Node) '
                'WHERE a.t IN [1,2] AND b.t IN [1,2] RETURN r.t AS t ORDER BY t',
                gid=self.gid if gid is None else gid)]

    def test_audit_probe_spider_fusion_and_full_reduce(self):
        for method, swap_colors, cross_types in product(
                ('spider_fusion', 'full_reduce'), (False, True), product((1, 2), repeat=2)):
            with self.subTest(method=method, swap_colors=swap_colors, wires=cross_types):
                g = sf_hadamard_parallel_via_fusion()
                cross_edges = [e for e in g.edges()
                               if {g.type(v) for v in g.edge_st(e)} == {1, 2}]
                for e, t in zip(cross_edges, cross_types):
                    g.set_edge_type(e, t)
                if swap_colors:
                    for v in g.vertices():
                        if g.type(v) in (1, 2):
                            g.set_type(v, 3 - g.type(v))
                self.load(g)
                getattr(self.db, method)(self.gid)
                self.assert_tensor_preserved(g)
                if method == 'spider_fusion':
                    self.assertEqual(self.internal_edge_types(), [] if cross_types == (1, 1) else [2])
                self.db.empty_graphdb(self.gid)

    def test_all_color_and_edge_multiplicities(self):
        # 4 ordered color pairs x 15 nonempty multiplicities = 60 cases.
        for colors, ns, nh in product(product((1, 2), repeat=2), range(4), range(4)):
            if ns + nh == 0:
                continue
            with self.subTest(colors=colors, simple=ns, hadamard=nh):
                g = parallel_pair(colors, ns, nh)
                self.load(g)
                self.db.remove_identities(self.gid)  # includes parallel cleanup
                same = colors[0] == colors[1]
                copying, cancelling = (ns, nh) if same else (nh, ns)
                copy_type, cancel_type = (1, 2) if same else (2, 1)
                expected = [copy_type] if copying else [cancel_type] * (cancelling % 2)
                self.assertEqual(self.internal_edge_types(), expected)
                self.assert_tensor_preserved(g)
                # A second cleanup is stable, including phases and wire types.
                with self.db.driver.session() as s:
                    before = s.run('MATCH(n:Node {graph_id:$gid}) '
                                   'RETURN id(n) AS id,n.phase AS p ORDER BY id', gid=self.gid).data()
                    self.db.remove_identities(self.gid)
                    after = s.run('MATCH(n:Node {graph_id:$gid}) '
                                  'RETURN id(n) AS id,n.phase AS p ORDER BY id', gid=self.gid).data()
                self.assertEqual(before, after)
                self.assertEqual(self.internal_edge_types(), expected)
                self.db.empty_graphdb(self.gid)

    def test_hopf_query_only_cancels_simple_edges_between_opposite_colors(self):
        for ns in (0, 1, 2, 3):
            with self.subTest(simple=ns):
                self.load(parallel_pair((1, 2), ns, 2))
                with self.db.driver.session() as s:
                    s.run(self.db._query('Hopf'), graph_id=self.gid).consume()
                self.assertEqual(self.internal_edge_types(), [1] * (ns % 2) + [2, 2])
                self.db.empty_graphdb(self.gid)

    def test_normalization_preserves_foreign_parallel_edges_and_phases(self):
        g = parallel_pair((1, 2), 3, 2)
        self.load(g)
        self.load(g, self.foreign)
        with self.db.driver.session() as s:
            query = ('MATCH(n:Node {graph_id:$gid}) '
                     'RETURN id(n) AS id, properties(n) AS p ORDER BY id')
            before = s.run(query, gid=self.foreign).data()
            self.db.spider_fusion(self.gid)
            self.assertEqual(s.run(query, gid=self.foreign).data(), before)
        self.assertEqual(self.internal_edge_types(self.foreign), [1, 1, 1, 2, 2])
        self.assertEqual(self.internal_edge_types(), [2])
        self.assert_tensor_preserved(g)


if __name__ == '__main__':
    unittest.main()
