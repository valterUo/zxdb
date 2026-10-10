"""Multi-instance focused design contracts, without database benchmarking."""
import unittest
from evaluation.research_workloads import build, digest, payload, make_plan, opportunities, rule_for
from evaluation.research_core import reference, validate_graph, verify
from evaluation.research_report import summarize


class FocusedTests(unittest.TestCase):
    def test_predeclared_grid_and_real_structural_variation(self):
        plan = make_plan('focused')
        self.assertEqual(plan, make_plan('focused'))
        self.assertEqual(len(plan['cases']), 610)
        families = {s['family'] for s in plan['cases']}
        self.assertEqual(len(families), 9)
        self.assertEqual(len({s['id'] for s in plan['cases']}), 610)
        for family in sorted(families):
            specs = [s for s in plan['cases'] if s['family'] == family and
                     (s['depth'] == 32 if family.startswith('random_') else s['size'] == 32)]
            self.assertEqual(len(specs), 10)
            graphs = [build(s) for s in specs]
            for graph in graphs:
                validate_graph(graph)
            self.assertEqual(len({digest(payload(g)) for g in graphs}), 10)
            self.assertEqual(payload(graphs[0]), payload(build(specs[0])))
            if family.startswith('copies_'):
                # Different vertex counts prove these are not just relabelings.
                self.assertEqual(len({g.num_vertices() for g in graphs}), 10)
                for spec in specs:
                    self.assertEqual(sum(spec['motif_counts'].values()), 32)
                    self.assertLessEqual(spec['motif_counts']['3'], 9)
                    self.assertGreater(opportunities(spec), 0)

    def test_each_motif_variant_rewrites_and_preserves_semantics(self):
        families = {s['family'] for s in make_plan('focused')['cases'] if s['family'].startswith('copies_')}
        for family in families:
            for size in (2, 3):
                with self.subTest(family=family, size=size):
                    graph = build(dict(family=family[7:], size=size, seed=0))
                    result = graph.copy()
                    reference(result, rule_for(family), float('inf'))
                    self.assertNotEqual(payload(graph), payload(result))
                    self.assertNotEqual(verify(graph, result)['status'], 'failed')

    def test_large_size_average_weights_sizes_equally_and_rejects_censoring(self):
        from evaluation.research_scale import large_scale_summary
        from evaluation.research_report import report
        import json
        import tempfile
        from pathlib import Path
        plan = make_plan('focused')
        plan['large_scale_summary']['families'] = ['copies_identity']
        # Deliberately unequal instance counts across sizes; each size still has equal weight.
        specs = []
        rows = []
        for i, size in enumerate((512, 1024, 2048)):
            selected = [s for s in plan['cases'] if s['family'] == 'copies_identity' and s['size'] == size][:i+1]
            specs.extend(selected)
            for spec in selected:
                rows.append(dict(kind='input', case=spec, sha256=spec['id'], metrics={'vertices': size*6}))
                for trial in range(2):
                    rows.append(dict(kind='trial', case=spec, trial=trial, warmup=False, status='ok',
                                     engines={e: dict(reduce_s=t, verification={'status':'verified'})
                                              for e, t in [('db', 1), ('pyzx', 2**(i+1))]}))
        plan['cases'] = specs
        result = large_scale_summary(plan, summarize(plan, rows), rows)[0]
        self.assertAlmostEqual(result['speedup'], 4)
        self.assertAlmostEqual(result['ci_low'], 4)
        self.assertEqual(result['distinct_inputs'], 6)
        self.assertEqual(result['measured_pairs'], 12)
        incomplete = rows[:-1]
        result = large_scale_summary(plan, summarize(plan, incomplete), incomplete)[0]
        self.assertFalse(result['complete'])
        self.assertIsNone(result['speedup'])
        self.assertAlmostEqual(result['completed_speedup'], 4)
        self.assertEqual(result['completed_pairs'], 11)
        # A wholly absent size changes the conditional population and is disclosed.
        partial = [r for r in rows if r['case']['size'] != 2048]
        result = large_scale_summary(plan, summarize(plan, partial), partial)[0]
        self.assertEqual(result['completed_sizes'], [512, 1024])
        self.assertAlmostEqual(result['completed_speedup'], 8**0.5)
        # Never combine unlike source versions or resource-limited observations.
        from copy import deepcopy
        mixed = deepcopy(rows)
        measured = [r for r in mixed if r['kind'] == 'trial']
        measured[0]['status'] = 'resource_limit'
        measured[1]['source_version'] = 'different'
        result = large_scale_summary(plan, summarize(plan, mixed), mixed)[0]
        self.assertEqual(result['completed_pairs'], 11)
        self.assertTrue(result['completed_mixed_versions'])
        self.assertIsNone(result['completed_speedup'])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'blocks').mkdir()
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'blocks'/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report(root)
            self.assertTrue((root/'figures'/'five_rules.tex').exists())
            self.assertIn('copies_identity.tikz', (root/'figures'/'five_rules.tex').read_text())
            table = (root/'large_scale_table.tex').read_text()
            self.assertIn('4.00', table)
            self.assertIn('conditional on completion', table)
            self.assertIn(r'95\% interval', table)
            self.assertNotIn(r'95\\% interval', table)
            self.assertEqual(json.loads((root/'large_scale_summary.json').read_text())[0]['distinct_inputs'], 6)
            saved_plan = (root/'plan.json').read_text()
            saved_curve = (root/'figures'/'copies_identity.dat').read_text()
            (root/'report-options.json').write_text(json.dumps({'exploratory_sizes': {'copies_identity': [2048]}}))
            report(root)
            self.assertIn('8.00', (root/'large_scale_table.tex').read_text())
            self.assertIn('Exploratory size selection', (root/'large_scale_table.tex').read_text())
            self.assertIn('4.00', (root/'large_scale_table_predeclared.tex').read_text())
            self.assertEqual(json.loads((root/'large_scale_summary.json').read_text())[0]['speedup'], 4)
            self.assertEqual((root/'plan.json').read_text(), saved_plan)
            self.assertEqual((root/'figures'/'copies_identity.dat').read_text(), saved_curve)
            (root/'report-options.json').write_text(json.dumps({'exploratory_sizes': {'copies_identity': [4096]}}))
            with self.assertRaisesRegex(ValueError, 'Exploratory sizes'):
                report(root)
            (root/'report-options.json').write_text(json.dumps({'table_scope': 'all_planned_sizes'}))
            report(root)
            table = (root/'all_cases_table.tex').read_text()
            self.assertIn('512, 1024, 2048', table)
            self.assertIn('4.00', table)
            self.assertNotIn('Exploratory size selection', table)
            self.assertIn('all_cases_table.tex', (root/'figures'/'figures.tex').read_text())
            self.assertEqual(json.loads((root/'large_scale_summary.json').read_text())[0]['speedup'], 4)



    def test_instance_counts_do_not_count_unrun_cases(self):
        plan = make_plan('focused'); plan['cases'] = plan['cases'][:10]
        summary = summarize(plan, [dict(kind='input', case=plan['cases'][i], sha256='same',
                                       metrics={'vertices': 10}) for i in (0, 1)])[0]
        self.assertEqual(summary['planned_seeds'], 10)
        self.assertEqual(summary['generated_instances'], 2)
        self.assertEqual(summary['distinct_inputs'], 1)
        self.assertEqual(summary['measured_instances'], 0)
        self.assertIsNone(summary['speedup'])


if __name__ == '__main__':
    unittest.main()
