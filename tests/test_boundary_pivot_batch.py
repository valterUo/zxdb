"""Boundary-pivot batching: semantic, overlap, ownership and planner checks."""
import os
import time
import unittest
from fractions import Fraction
from itertools import product

import pyzx as zx
from evaluation.cases import RULES, pb_basic
from evaluation.harness import _tensors_equal
from evaluation.research_core import load, read, verify, reference, compare_outputs, CountingDriver
from evaluation.research_workloads import build, payload
from tests.test_pivot_materialization import pivot_neighborhood
import tests.test_research as research_tests


def boundary_neighborhood(groups=(True,True,True), existing=False, j_phase=1, a_phase=Fraction(1,4)):
    g=pivot_neighborhood(groups,existing)
    g.set_phase(0,j_phase);g.set_phase(1,a_phase)
    bd=g.add_vertex(zx.VertexType.BOUNDARY)
    g.add_edge((1,bd));g.set_outputs(g.outputs()+(bd,))
    return g


def overlapping_boundaries(shared_alpha=False):
    g=zx.Graph();centers=[];alphas=[];neighbors=[];outputs=[]
    for i in range(2):
        j=g.add_vertex(zx.VertexType.Z,phase=i);centers.append(j)
        if shared_alpha and i:
            a=alphas[0]
        else:
            a=g.add_vertex(zx.VertexType.Z,phase=Fraction(1,4));alphas.append(a)
            b=g.add_vertex(zx.VertexType.BOUNDARY);outputs.append(b);g.add_edge((a,b))
        g.add_edge((j,a),zx.EdgeType.HADAMARD)
        n=g.add_vertex(zx.VertexType.Z,phase=Fraction(3,4));neighbors.append(n)
        b=g.add_vertex(zx.VertexType.BOUNDARY);outputs.append(b)
        g.add_edge((j,n),zx.EdgeType.HADAMARD);g.add_edge((n,b))
    if not shared_alpha:
        g.add_edge((centers[1],neighbors[0]),zx.EdgeType.HADAMARD)
    g.set_outputs(tuple(outputs))
    return g


@unittest.skipUnless(os.environ.get('ZXDB_RESEARCH_LIVE')=='1','Opt-in Memgraph checks')
class BoundaryPivotBatchTests(unittest.TestCase):
    setUp=research_tests.ResearchLiveTests.setUp
    tearDown=research_tests.ResearchLiveTests.tearDown

    def batch(self):
        return self.db.apply_rule_optimized('pivot_boundary_rule',self.gid,exclusive_access=True)

    def test_existing_adversarial_cases(self):
        for name,builder,*_ in RULES['pivot_boundary']['cases']:
            with self.subTest(case=name):
                g=builder();load(self.db,g,self.gid);self.batch();out=read(self.db,self.gid)
                self.assertTrue(_tensors_equal(g,out))
                reference(g,'pivot_boundary',time.monotonic()+10)
                self.assertTrue(_tensors_equal(g,out))
                self.assertEqual(self.batch(),0)

    def test_neighbor_groups_edge_toggles_and_phases(self):
        for groups,existing,j_phase,a_phase in product(product((False,True),repeat=3),
                                                      (False,True),(0,1),(0,Fraction(1,4),1,Fraction(3,2))):
            with self.subTest(groups=groups,existing=existing,j=j_phase,a=a_phase):
                g=boundary_neighborhood(groups,existing,j_phase,a_phase)
                load(self.db,g,self.gid)
                self.assertEqual(self.batch(),1)
                out=read(self.db,self.gid)
                self.assertTrue(_tensors_equal(g,out))
                self.assertEqual(self.batch(),0)

    def test_shared_neighbor_or_boundary_owner_conflicts(self):
        for shared_alpha in (False,True):
            with self.subTest(shared_alpha=shared_alpha):
                g=overlapping_boundaries(shared_alpha);load(self.db,g,self.gid)
                with self.db.driver.session() as session:
                    count=self.db._run_count(session,self.db._query('Pivot boundary - batched'),
                                             'pivot_operations_performed',self.gid)
                self.assertEqual(count,1)
                self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
                self.batch();out=read(self.db,self.gid)
                self.assertEqual(verify(g,out)['status'],'verified')
                reference(g,'pivot_boundary',time.monotonic()+10)
                self.assertTrue(_tensors_equal(g,out))

    def test_invalid_candidates_unchanged(self):
        for invalid in ('second_boundary','hadamard_boundary','nonpauli_j','simple_interior','leaf_neighbor','boundary_degree'):
            with self.subTest(invalid=invalid):
                g=boundary_neighborhood((True,False,False));bd=g.outputs()[-1]
                # Keep the other neighbor from being a separate, valid boundary
                # candidate while invalidating the intended pair (0, 1).
                g.set_edge_type(g.edge(2,3),zx.EdgeType.HADAMARD)
                if invalid=='second_boundary':
                    b=g.add_vertex(zx.VertexType.BOUNDARY);g.add_edge((1,b));g.set_outputs(g.outputs()+(b,))
                elif invalid=='hadamard_boundary':g.set_edge_type(g.edge(1,bd),zx.EdgeType.HADAMARD)
                elif invalid=='nonpauli_j':g.set_phase(0,Fraction(1,4))
                elif invalid=='simple_interior':g.set_edge_type(g.edge(0,1),zx.EdgeType.SIMPLE)
                elif invalid=='leaf_neighbor':
                    n=g.add_vertex(zx.VertexType.Z,phase=Fraction(1,4));g.add_edge((0,n),zx.EdgeType.HADAMARD)
                else:
                    n=g.add_vertex(zx.VertexType.Z,phase=Fraction(1,4));g.add_edge((bd,n))
                load(self.db,g,self.gid);before=payload(read(self.db,self.gid))
                self.assertEqual(self.batch(),0)
                self.assertEqual(before,payload(read(self.db,self.gid)))

    def test_mixed_copies_three_submissions_and_foreign_graph_unchanged(self):
        g=build(dict(family='copies_pivot_boundary',size=24,seed=2,motif_counts={'2':6,'3':18}))
        load(self.db,g,self.gid);foreign=self.gid+'_other';load(self.db,pb_basic(),foreign)
        before=payload(read(self.db,foreign));counter=CountingDriver(self.db.driver);self.db._driver=counter
        try:
            self.assertEqual(self.batch(),24)
            self.assertEqual(counter.submissions,3)
            out=read(self.db,self.gid)
            self.assertEqual(payload(read(self.db,foreign)),before)
            reference(g,'pivot_boundary',time.monotonic()+10)
            self.assertEqual(compare_outputs(out,g,max_iso_vertices=1024,iso_timeout_s=5)['graph']['status'],'equal')
        finally:
            self.db._driver=counter.driver;self.db.empty_graphdb(foreign)

    def test_benchmark_motifs_tensor_and_neighbor_plan(self):
        for size in (2,3):
            g=build(dict(family='pivot_boundary',size=size,seed=0))
            load(self.db,g,self.gid);self.batch();self.assertTrue(_tensors_equal(g,read(self.db,self.gid)))
        load(self.db,build(dict(family='copies_pivot_boundary',size=8,seed=0)),self.gid)
        with self.db.driver.session() as session:
            plan=str(session.run('EXPLAIN '+self.db._query('Pivot boundary - batched'),graph_id=self.gid).data())
        self.assertNotIn('ScanAll (nj)',plan);self.assertNotIn('ScanAll (na)',plan)

    def test_optimized_full_reduce_small_circuits(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                g=build(dict(family='random_clifford_t',size=2,depth=3,seed=seed))
                load(self.db,g,self.gid);self.db.full_reduce_optimized(self.gid,exclusive_access=True)
                self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')

    def test_pauli_boundary_neighbor_groups_phases_and_both_wire_types(self):
        for groups,existing,j_phase,a_phase,wire in product(product((False,True),repeat=3),
                                                         (False,True),(0,1),(0,1),(1,2)):
            with self.subTest(groups=groups,existing=existing,j=j_phase,a=a_phase,wire=wire):
                g=boundary_neighborhood(groups,existing,j_phase,a_phase)
                g.set_edge_type(g.edge(1,g.outputs()[-1]),zx.EdgeType(wire))
                load(self.db,g,self.gid)
                with self.db.driver.session() as session:
                    count=self.db._run_count(session,self.db._query('Pivot boundary Pauli - batched'),
                                             'interior_pauli_removed',self.gid)
                self.assertEqual(count,1)
                self.assertTrue(_tensors_equal(g,read(self.db,self.gid)))

    def test_pauli_batches_conflict_and_disjoint_selection(self):
        for shared_alpha in (False,True):
            g=overlapping_boundaries(shared_alpha)
            for v in g.vertices():
                if g.type(v)==zx.VertexType.Z and g.phase(v)==Fraction(1,4):
                    g.set_phase(v,0)
            load(self.db,g,self.gid)
            with self.db.driver.session() as session:
                count=self.db._run_count(session,self.db._query('Pivot boundary Pauli - batched'),
                                         'interior_pauli_removed',self.gid)
            self.assertEqual(count,1)
            self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
        g=build(dict(family='copies_pivot_boundary',size=16,seed=0))
        # The alpha centers have degree four and one boundary neighbor;
        # peripheral quarter-phase spiders have degree two.
        for v in g.vertices():
            if g.type(v)==zx.VertexType.Z and g.vertex_degree(v)==4:
                g.set_phase(v,0)
        load(self.db,g,self.gid)
        self.assertEqual(self.db.apply_rule_optimized('pivot_rule',self.gid,exclusive_access=True),16)
        out=read(self.db,self.gid)
        reference(g,'pivot',time.monotonic()+10)
        self.assertEqual(compare_outputs(out,g,max_iso_vertices=1024,iso_timeout_s=5)['graph']['status'],'equal')
