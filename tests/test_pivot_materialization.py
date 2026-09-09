"""Semantic and liveness regressions for materialized pivot edge toggles."""
import os
import unittest
import uuid
from fractions import Fraction
from itertools import product

import pyzx as zx
from neo4j import GraphDatabase
from evaluation.research_core import load, read, verify, validate_graph
from evaluation.research_workloads import build
from zxdb.zxdb import ZXdb


def pivot_neighborhood(groups, existing):
    g = zx.Graph()
    a = g.add_vertex(zx.VertexType.Z)
    b = g.add_vertex(zx.VertexType.Z, phase=1)
    g.add_edge((a, b), zx.EdgeType.HADAMARD)
    leaves = []
    outputs = []
    for group, enabled in enumerate(groups):
        if not enabled:
            continue
        v = g.add_vertex(zx.VertexType.Z, phase=Fraction(1, 4))
        if group in (0, 2):
            g.add_edge((a, v), zx.EdgeType.HADAMARD)
        if group in (1, 2):
            g.add_edge((b, v), zx.EdgeType.HADAMARD)
        boundary = g.add_vertex(zx.VertexType.BOUNDARY)
        g.add_edge((v, boundary)); outputs.append(boundary); leaves.append(v)
    if existing:
        for i, u in enumerate(leaves):
            for v in leaves[i+1:]:
                g.add_edge((u, v), zx.EdgeType.HADAMARD)
    g.set_outputs(tuple(outputs))
    return g


@unittest.skipUnless(os.environ.get('ZXDB_RESEARCH_LIVE') == '1', 'Opt-in Memgraph checks')
class PivotMaterializationTests(unittest.TestCase):
    def setUp(self):
        uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
        auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            with driver.session() as s:
                if s.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                    self.skipTest('Requires a dedicated empty database')
        self.db = ZXdb(uri=uri, user=auth[0], password=auth[1])
        self.gid = 'pivot_materialization_' + uuid.uuid4().hex

    def tearDown(self):
        self.db.empty_graphdb(self.gid); self.db.close()

    def test_all_neighbor_groups_with_existing_and_absent_edges(self):
        for groups, existing in product(product((False, True), repeat=3), (False, True)):
            with self.subTest(groups=groups, existing=existing):
                original = pivot_neighborhood(groups, existing)
                load(self.db, original, self.gid)
                self.assertEqual(self.db.pivot_rule(self.gid), 1)
                result = read(self.db, self.gid)
                self.assertEqual(verify(original, result)['status'], 'verified')
                self.assertEqual(self.db.pivot_rule(self.gid), 0)

    def test_reported_circuit_repeated_full_reduce_liveness(self):
        # A crash regression, not a large-circuit semantic proof. Tensor checking
        # this 32-open-leg map exceeds the research oracle's resource bound.
        graph = build(dict(family='random_t_heavy', size=16, depth=80, seed=4))
        for repetition in range(6):
            with self.subTest(repetition=repetition):
                load(self.db, graph, self.gid)
                self.db.full_reduce(self.gid)
                result = read(self.db, self.gid)
                validate_graph(result)
                self.assertEqual((len(result.inputs()), len(result.outputs())), (16, 16))


if __name__ == '__main__':
    unittest.main()
