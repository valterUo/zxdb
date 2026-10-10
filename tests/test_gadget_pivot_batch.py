"""Gadget-pivot batch regressions; live checks use an isolated Memgraph."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from fractions import Fraction
from itertools import product
from unittest.mock import patch

import pyzx as zx
from evaluation.cases import RULES, pg_basic
from evaluation.research_core import load, read, verify, reference, compare_outputs, CountingDriver
from evaluation.research_workloads import build, payload
from tests.test_pivot_materialization import pivot_neighborhood
import tests.test_research as research_tests
from evaluation.harness import _tensors_equal


class GadgetPivotProbeTests(unittest.TestCase):
    def test_timeout_with_dead_database_keeps_worker_diagnostics(self):
        from evaluation.gadget_pivot_probe import main
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'probe.json'
            with patch('sys.argv',['probe','--output',str(path),'--budget-seconds','30',
                                   '--rule','pivot_boundary']), \
                 patch.dict(os.environ,{'ZXDB_URI':'bolt://unused:7688'}), \
                 patch('evaluation.gadget_pivot_probe.subprocess.run',
                       side_effect=subprocess.TimeoutExpired('worker',30,stderr=b'disconnected in worker')), \
                 patch('evaluation.gadget_pivot_probe.ZXdb',side_effect=ConnectionError('server down')):
                main()
            report=json.loads(path.read_text())
            self.assertEqual(report['rule'],'pivot_boundary')
            self.assertEqual(report['status'],'timeout_and_cleanup_failed')
            self.assertEqual(report['worker_stderr'],'disconnected in worker')
            self.assertIn('server down',report['cleanup_error'])

    def test_hard_timeout_preserves_report_and_cleans_only_worker_graph(self):
        from evaluation.gadget_pivot_probe import main
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'probe.json'
            with patch('sys.argv',['probe','--output',str(path),'--budget-seconds','30']), \
                 patch.dict(os.environ,{'ZXDB_URI':'bolt://unused:7688'}), \
                 patch('evaluation.gadget_pivot_probe.subprocess.run',
                       side_effect=subprocess.TimeoutExpired('worker',30)) as run, \
                 patch('evaluation.gadget_pivot_probe.ZXdb') as database:
                main()
            report=json.loads(path.read_text())
            self.assertEqual(report['status'],'budget_exhausted')
            self.assertEqual(report['cases'],[])
            gid=run.call_args.args[0][-1]
            self.assertTrue(gid.startswith('gadget_probe_'))
            database.return_value.__enter__.return_value.empty_graphdb.assert_called_once_with(gid)


def overlapping_pivots(shared=False):
    g = zx.Graph()
    centers=[]; tips=[]; outputs=[]
    for _ in range(2):
        j=g.add_vertex(zx.VertexType.Z, phase=1)
        a=g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
        g.add_edge((j,a),zx.EdgeType.HADAMARD); centers.append(j)
        for v in (j,a):
            n=g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4)); tips.append(n)
            b=g.add_vertex(zx.VertexType.BOUNDARY);outputs.append(b)
            g.add_edge((v,n),zx.EdgeType.HADAMARD);g.add_edge((n,b))
    g.add_edge((centers[1], tips[0] if shared else centers[0]),zx.EdgeType.HADAMARD)
    g.set_outputs(tuple(outputs))
    return g


@unittest.skipUnless(os.environ.get('ZXDB_RESEARCH_LIVE')=='1','Opt-in Memgraph checks')
class GadgetPivotBatchTests(unittest.TestCase):
    setUp=research_tests.ResearchLiveTests.setUp
    tearDown=research_tests.ResearchLiveTests.tearDown

    def batch(self):
        return self.db.apply_rule_optimized('pivot_gadget_rule',self.gid,exclusive_access=True)

    def test_existing_adversarial_cases(self):
        for name,builder,*_ in RULES['pivot_gadget']['cases']:
            with self.subTest(case=name):
                g=builder();load(self.db,g,self.gid);self.batch();out=read(self.db,self.gid)
                # Historical fixtures also include two-legged H boxes.
                if any(int(t) == 3 for t in g.types().values()):
                    self.assertTrue(_tensors_equal(g,out))
                else:
                    self.assertEqual(verify(g,out)['status'],'verified')
                reference(g,'pivot_gadget',time.monotonic()+10)
                self.assertTrue(_tensors_equal(out,g))
                self.assertEqual(self.batch(),0)

    def test_all_neighbor_groups_edge_toggles_and_phases(self):
        for groups,existing,phase in product(product((False,True),repeat=3),(False,True),(0,1)):
            with self.subTest(groups=groups,existing=existing,phase=phase):
                g=pivot_neighborhood(groups,existing)
                g.set_phase(0,phase);g.set_phase(1,Fraction(3,4))
                load(self.db,g,self.gid);self.batch()
                self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
                self.assertEqual(self.batch(),0)

    def test_both_benchmark_motifs_and_mixed_components(self):
        for size in (2,3):
            with self.subTest(motif_size=size):
                g=build(dict(family='pivot_gadget',size=size,seed=0))
                load(self.db,g,self.gid);self.batch()
                self.assertTrue(_tensors_equal(g,read(self.db,self.gid)))
        g=build(dict(family='copies_pivot_gadget',size=24,seed=2,
                     motif_counts={'2':6,'3':18}))
        load(self.db,g,self.gid)
        self.assertEqual(self.batch(),24)
        out=read(self.db,self.gid)
        reference(g,'pivot_gadget',time.monotonic()+10)
        self.assertEqual(compare_outputs(out,g,max_iso_vertices=1024,iso_timeout_s=5)['graph']['status'],'equal')

    def test_overlapping_matches_are_not_selected_together(self):
        for shared in (False,True):
            g=overlapping_pivots(shared);load(self.db,g,self.gid)
            with self.db.driver.session() as session:
                n=self.db._run_count(session,self.db._query('Pivot gadget - batched'),'pivot_operations_performed',self.gid)
            self.assertEqual(n,1)
            self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
            self.batch()
            self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
            self.assertEqual(self.db.pivot_gadget_rule(self.gid),0)

    def test_disjoint_instances_batch_and_preserve_foreign_graph(self):
        g=build(dict(family='copies_pivot_gadget',size=32,seed=0))
        load(self.db,g,self.gid);foreign=self.gid+'_other'
        load(self.db,pg_basic(),foreign);before=payload(read(self.db,foreign))
        counter=CountingDriver(self.db.driver);self.db._driver=counter
        try:
            self.assertEqual(self.batch(),32)
            self.assertEqual(counter.submissions,3) # ownership, batch, fixed-point check
            out=read(self.db,self.gid)
            self.assertEqual(payload(read(self.db,foreign)),before)
            reference(g,'pivot_gadget',time.monotonic()+10)
            self.assertEqual(compare_outputs(out,g,max_iso_vertices=1024,iso_timeout_s=5)['graph']['status'],'equal')
        finally:
            self.db._driver=counter.driver;self.db.empty_graphdb(foreign)

    def test_optimized_full_reduce_small_circuits(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                g=build(dict(family='random_clifford_t',size=2,depth=3,seed=seed))
                load(self.db,g,self.gid);self.db.full_reduce_optimized(self.gid,exclusive_access=True)
                self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')

    def test_neighbor_plan_expands_bound_centers(self):
        load(self.db,build(dict(family='copies_pivot_gadget',size=8,seed=0)),self.gid)
        with self.db.driver.session() as session:
            plan=session.run('EXPLAIN '+self.db._query('Pivot gadget - batched'),
                             graph_id=self.gid).data()
        operators=str(plan)
        self.assertNotIn('ScanAll (nj)',operators)
        self.assertNotIn('ScanAll (na)',operators)

    def test_boundary_candidate_keeps_its_own_boundary_and_phases(self):
        # These inputs exposed incorrect values carried through the old sorted
        # boundary-candidate projection in Memgraph 3.6.2. Vary allocation order.
        for repeat,seed in product(range(4),(1,10)):
            with self.subTest(repeat=repeat,seed=seed):
                g=build(dict(family='random_clifford_t',size=2,depth=3,seed=seed))
                load(self.db,g,self.gid);self.db.full_reduce(self.gid)
                self.assertEqual(verify(g,read(self.db,self.gid))['status'],'verified')
