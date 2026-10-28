"""Live semantics and isolation checks for the native full-reduction path."""
import os
import unittest
import uuid
from itertools import product
from fractions import Fraction

import pyzx as zx

from evaluation.cases import RULES, sf_chain3, lc_toggle_off
from evaluation.circuits import build_suite
from evaluation.core import CountingDriver, load, read, verify
from evaluation.workloads import payload, build
from zxdb.zxdb import ZXdb


@unittest.skipUnless(os.environ.get('ZXDB_LIVE') == '1', 'Requires compiled zxqmcpp module')
class NativeReductionTests(unittest.TestCase):
    def setUp(self):
        self.db = ZXdb(uri=os.environ.get('ZXDB_URI', 'bolt://localhost:7687'),
                       user=os.environ.get('ZXDB_USER', ''), password=os.environ.get('ZXDB_PASSWORD', ''))
        self.db._driver = CountingDriver(self.db.driver, timeout=30)
        self.gid = 'native_test_' + uuid.uuid4().hex
        self.foreign = self.gid + '_foreign'
        load(self.db, sf_chain3(), self.foreign)
        self.foreign_before = payload(read(self.db, self.foreign))

    def tearDown(self):
        self.assertEqual(payload(read(self.db, self.foreign)), self.foreign_before)
        self.db.empty_graphdb(self.gid)
        self.db.empty_graphdb(self.foreign)
        self.db.close()

    def native(self, name):
        with self.db.driver.session() as s:
            return s.run(f'CALL zxqmcpp.{name}($gid) YIELD applied RETURN applied',
                         gid=self.gid).single()['applied']

    def test_individual_rule_fixtures(self):
        for rule, procedure in [('pivot', 'pivot'), ('lcomp', 'lcomp'), ('pivot_gadget', 'gadget'),
                                ('supplementarity', 'supplementarity'), ('pivot_boundary', 'boundary')]:
            for name, builder, *_ in RULES[rule]['cases']:
                with self.subTest(rule=rule, case=name):
                    g = builder()
                    try:
                        load(self.db, g, self.gid)
                        count = self.native(procedure)
                        self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
                        self.assertEqual(self.native(procedure), 0)
                    except Exception:
                        print(f"Last database query for {rule}/{name}: {self.db.driver.last_query}", flush=True)
                        raise
                    if name in ('basic', 'basic_00', 'basic_type1', 'type2_sum_zero'):
                        self.assertGreater(count, 0)

    def test_pauli_boundary_neighbor_groups_phases_and_wire_types(self):
        from tests.test_boundary_pivot_batch import boundary_neighborhood
        for groups, existing, pa, pb, bt in product(product((False, True), repeat=3),
                                                    (0, 1, 2), (0, 1), (0, 1), (1, 2)):
            with self.subTest(groups=groups, existing=existing, pa=pa, pb=pb, bt=bt):
                g = boundary_neighborhood(groups, bool(existing), pa, pb)
                boundary = g.outputs()[-1]
                g.set_edge_type(g.edge(1, boundary), zx.EdgeType(bt))
                if existing == 1:
                    for edge in list(g.edges()):
                        a, b = g.edge_st(edge)
                        if a not in (0, 1) and b not in (0, 1) and g.type(a) == g.type(b) == zx.VertexType.Z:
                            g.set_edge_type(edge, zx.EdgeType.SIMPLE)
                load(self.db, g, self.gid)
                self.assertEqual(self.native('pauli_boundary'), 1)
                self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(self.native('pauli_boundary'), 0)

    def test_gadget_boundary_neighbor_groups_and_phases(self):
        from tests.test_boundary_pivot_batch import boundary_neighborhood
        for groups, existing, pa, pb in product(product((False, True), repeat=3),
                                                 (False, True), (0, 1),
                                                 (0, Fraction(1, 4), 1, Fraction(3, 2))):
            with self.subTest(groups=groups, existing=existing, pa=pa, pb=pb):
                g = boundary_neighborhood(groups, existing, pa, pb)
                load(self.db, g, self.gid)
                self.assertEqual(self.native('boundary'), 1)
                self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
                self.assertEqual(self.native('boundary'), 0)

    def test_pauli_boundary_overlaps_and_invalid_candidates(self):
        from tests.test_boundary_pivot_batch import boundary_neighborhood, overlapping_boundaries
        for shared in (False, True):
            g = overlapping_boundaries(shared)
            for v in g.vertices():
                if g.phase(v) == Fraction(1, 4): g.set_phase(v, 1)
            load(self.db, g, self.gid)
            self.assertGreater(self.native('pauli_boundary'), 0)
            self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
        for invalid in ('nonpauli', 'two_boundaries', 'simple_interior'):
            g = boundary_neighborhood(a_phase=0)
            if invalid == 'nonpauli': g.set_phase(1, Fraction(1, 4))
            elif invalid == 'simple_interior': g.set_edge_type(g.edge(0, 1), zx.EdgeType.SIMPLE)
            else:
                v = g.add_vertex(zx.VertexType.BOUNDARY)
                g.add_edge((1, v)); g.set_outputs(g.outputs() + (v,))
            load(self.db, g, self.gid)
            before = payload(read(self.db, self.gid))
            self.assertEqual(self.native('pauli_boundary'), 0)
            self.assertEqual(payload(read(self.db, self.gid)), before)

    def test_supplementarity_batches_and_rejects_simple_incident_edges(self):
        for size in (2, 3):
            g = build(dict(family='copies_supplementarity', size=size, seed=0))
            load(self.db, g, self.gid)
            self.assertGreater(self.native('supplementarity'), 0)
            self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')
            self.assertEqual(self.native('supplementarity'), 0)
        g = RULES['supplementarity']['cases'][0][1]()
        for edge in list(g.edges()):
            g.set_edge_type(edge, zx.EdgeType.SIMPLE)
        load(self.db, g, self.gid)
        before = payload(read(self.db, self.gid))
        self.assertEqual(self.native('supplementarity'), 0)
        self.assertEqual(payload(read(self.db, self.gid)), before)

    def test_simple_neighbor_edge_is_preserved_by_hadamard_toggle(self):
        g = lc_toggle_off()
        center = next(v for v in g.vertices() if g.phase(v) == .5)
        edge = next(e for e in g.edges() if center not in g.edge_st(e)
                    and all(g.type(v) == zx.VertexType.Z for v in g.edge_st(e)))
        g.set_edge_type(edge, zx.EdgeType.SIMPLE)
        load(self.db, g, self.gid)
        self.native('lcomp')
        self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')

    def test_full_reduce_preserves_zero_factors(self):
        from evaluate import zero_cases, reduce_database
        for variant in ('control', 'optimized', 'native'):
            for name, g in zero_cases():
                with self.subTest(variant=variant, case=name):
                    load(self.db, g, self.gid)
                    reduce_database(self.db, 'full_reduce', self.gid, variant)
                    self.assertEqual(verify(g, read(self.db, self.gid))['status'], 'verified')

    def test_full_reduce_fixtures(self):
        for name, builder in build_suite():
            with self.subTest(case=name):
                g = builder()
                load(self.db, g, self.gid)
                self.db.full_reduce_native(self.gid)
                check = verify(g, read(self.db, self.gid))
                self.assertNotEqual(check['status'], 'failed', check)
                if not name.startswith(('wide_q10', 'wide_q12')):
                    self.assertEqual(check['status'], 'verified', check)

    def test_direct_call_rejects_cross_graph_edges_before_mutation(self):
        load(self.db, sf_chain3(), self.gid)
        before = payload(read(self.db, self.gid))
        with self.db.driver.session() as s:
            s.run('MATCH(a:Node {graph_id:$gid}), (b:Node {graph_id:$other}) '
                  'WITH a,b LIMIT 1 CREATE(a)-[:Wire {t:2}]->(b)',
                  gid=self.gid, other=self.foreign).consume()
        for name in ('pivot', 'lcomp', 'gadget', 'supplementarity', 'pauli_boundary', 'boundary'):
            with self.assertRaisesRegex(Exception, 'Cross-graph'):
                self.native(name)
            self.assertEqual(payload(read(self.db, self.gid)), before)


if __name__ == '__main__':
    unittest.main()
