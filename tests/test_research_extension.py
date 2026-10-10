"""Append-only focused campaign resume, without a live database."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from evaluation.research import run
from evaluation.research_adaptive import ordered_cases
from evaluation.research_extend import extended_plan, compatible_sources, AUDITED_SOURCE_UPDATES
from evaluation.research_workloads import make_plan
from evaluation.research_recovery import append, records


class ExtensionTests(TestCase):
    def test_append_keeps_existing_cases_indices_order_and_settings(self):
        all_cases = make_plan('focused')
        selected = dict(all_cases, cases=[s for s in all_cases['cases'] if s['family'] in
                         ('copies_identity', 'copies_lcomp', 'copies_spider_fusion', 'copies_pivot', 'copies_gadget_fusion')])
        extended = extended_plan(selected, all_cases)
        self.assertEqual(len(extended['cases']), 610)
        self.assertEqual(extended['cases'][:350], selected['cases'])
        self.assertEqual(ordered_cases(extended)[:350], ordered_cases(selected))
        self.assertEqual(sorted(ordered_cases(extended)), list(range(610)))
        self.assertEqual(extended_plan(extended, all_cases), extended)
        for key in selected:
            if key != 'cases': self.assertEqual(selected[key], extended[key])

    def test_compatibility_is_limited_to_report_and_exact_audited_updates(self):
        name = 'evaluation/research.py'; before, after = AUDITED_SOURCE_UPDATES[name][0]
        saved = {name: before, 'evaluation/research_core.py': 'fixed'}
        current = {name: after, 'evaluation/research_core.py': 'fixed', 'evaluation/research_scale.py': 'new'}
        self.assertTrue(compatible_sources(saved, current))
        current['evaluation/research_core.py'] = 'different'
        self.assertFalse(compatible_sources(saved, current))
        current['evaluation/research_core.py'] = 'fixed'; current[name] = 'unaudited'
        self.assertFalse(compatible_sources(saved, current))

    def test_resumed_extension_runs_only_new_case_and_keeps_measurement_version(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); output=root/'results'; path=root/'plan.json'
            candidate = make_plan('focused')
            one = next(s for s in candidate['cases'] if s['family']=='copies_identity')
            new = next(s for s in candidate['cases'] if s['family']=='copies_bialgebra')
            plan = dict(candidate, cases=[one]); path.write_text(json.dumps(plan))
            args = SimpleNamespace(plan=str(path), output=str(output), budget_seconds=1000)
            before, after = AUDITED_SOURCE_UPDATES['evaluation/research.py'][0]
            env = {'source_hashes': {'evaluation/research.py':before}}
            calls=[]; versions=[]
            def child(command, **kwargs):
                index=int(command[command.index('--index')+1]); spec=plan['cases'][index]
                calls.append(spec['id']);versions.append(kwargs['env']['ZXDB_SOURCE_VERSION'])
                block=output/'blocks'/(spec['id']+'.jsonl')
                for trial in range(2):
                    append(block,dict(kind='trial',case=spec,trial=trial,warmup=False,status='ok',
                                      source_version=versions[-1],engines={e:dict(reduce_s=t,verification={'status':'verified'})
                                                                         for e,t in [('db',1),('pyzx',2)]}))
                return SimpleNamespace(returncode=0)
            with patch('evaluation.research.environment', side_effect=lambda:env), \
                 patch('evaluation.research.subprocess.run',side_effect=child), \
                 patch('evaluation.research_recovery.ready',return_value=True), \
                 patch('evaluation.research_recovery.cleanup_owned'):
                run(args)
                block=output/'blocks'/(one['id']+'.jsonl'); original=block.read_bytes()
                plan=extended_plan(plan,dict(candidate,cases=[one,new]))
                path.write_text(json.dumps(plan));(output/'plan.json').write_text(json.dumps(plan))
                env={'source_hashes':{'evaluation/research.py':after}}
                run(args)
                self.assertEqual(calls,[one['id'],new['id']])
                self.assertEqual(versions[0],versions[1])
                self.assertEqual(block.read_bytes(),original)
                run(args)
                self.assertEqual(len(calls),2)
