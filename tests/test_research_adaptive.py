"""Adaptive ordering, timeout evidence and resumed campaign contracts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.research import run
from evaluation.research_adaptive import ordered_cases, series_key, scale_value, stop_evidence, timed_out, blocked_predecessor
from evaluation.research_workloads import make_plan
from evaluation.research_recovery import append, records


class AdaptiveTests(unittest.TestCase):
    def plan(self):
        plan = make_plan('adaptive')
        plan['cases'] = [s for s in plan['cases'] if s['family'] == 'copies_pivot'][:5]
        return plan

    def test_ladders_are_deterministic_ascending_and_start_small(self):
        plan = make_plan('adaptive')
        order = ordered_cases(plan)
        self.assertEqual(order, ordered_cases(plan))
        self.assertEqual(sorted(order), list(range(len(plan['cases']))))
        groups = {}
        for i in order:
            spec = plan['cases'][i]
            groups.setdefault(series_key(spec), []).append(scale_value(spec))
        for key, sizes in groups.items():
            self.assertEqual(sizes, sorted(sizes))
            self.assertEqual(sizes[0], 4 if key[0].startswith(('random_', 'layered_')) else 2)
        self.assertEqual(max(s['size'] for s in plan['cases'] if s['family'] == 'pivot'), 512)
        self.assertGreater(len({series_key(plan['cases'][i]) for i in order[:10]}), 5)

    def test_only_repeated_both_engine_timeouts_stop_growth(self):
        plan = self.plan(); specs = plan['cases']
        both = [dict(kind='limit_stop', reason='both_engines_timeout')]
        one = [dict(kind='trial', trial=i, warmup=False,
                    engines=dict(db=dict(status='ok'), pyzx=dict(status='timeout'))) for i in range(2)]
        histories = {s['id']: both for s in specs[:3]}
        self.assertEqual(stop_evidence(plan, specs[3], histories), [s['id'] for s in specs[:3]])
        histories[specs[1]['id']] = one
        self.assertFalse(stop_evidence(plan, specs[4], histories))
        self.assertFalse(timed_out(one, 2))
        self.assertFalse(timed_out([dict(kind='timeout')], 2))
        self.assertFalse(timed_out([dict(kind='error')], 2))
        histories[specs[1]['id']] = [dict(kind='case_budget')]
        self.assertEqual(blocked_predecessor(plan, specs[2], histories), [specs[1]['id']])

    def test_campaign_stops_after_three_and_resume_preserves_decisions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); plan = self.plan(); output = root/'results'
            plan_path = root/'plan.json'; plan_path.write_text(json.dumps(plan))
            args = SimpleNamespace(plan=str(plan_path), output=str(output), budget_seconds=1000)
            calls = []
            def child(command, **kwargs):
                index = int(command[command.index('--index')+1]); spec = plan['cases'][index]
                calls.append(index)
                self.assertLessEqual(kwargs['timeout'], 120)
                self.assertEqual(float(kwargs['env']['ZXDB_BLOCK_TIMEOUT_S']), kwargs['timeout'])
                append(output/'blocks'/(spec['id']+'.jsonl'), dict(case=spec, kind='limit_stop', reason='both_engines_timeout'))
                return SimpleNamespace(returncode=0)
            with patch('evaluation.research.environment', return_value=dict(source_hashes={'test':'v1'})), \
                 patch('evaluation.research.subprocess.run', side_effect=child), \
                 patch('evaluation.research_recovery.ready', return_value=True), \
                 patch('evaluation.research_recovery.cleanup_owned'):
                run(args)
                self.assertEqual(calls, [0, 1, 2])
                run(args)
                self.assertEqual(calls, [0, 1, 2])
                outcomes = json.loads((output/'campaign.json').read_text())['outcomes']
                self.assertEqual([o['status'] for o in outcomes], ['both_engines_timeout']*3+['not_run_adaptive_stop']*2)
                summary = json.loads((output/'summary.json').read_text())
                self.assertTrue(all(s['speedup'] is None for s in summary))
                self.assertEqual(sum(s['adaptive_skip_cases'] for s in summary), 2)
                self.assertTrue((output/'figures'/'paper_rules.tex').exists())
                args.case_timeout_seconds = 60
                with self.assertRaisesRegex(ValueError, 'changing adaptive time limits'):
                    run(args)

    def test_worker_timeout_skips_redundant_attempts_and_preserves_other_engine(self):
        from evaluation.research import worker
        from evaluation.research_workloads import build
        from unittest.mock import Mock
        for db_timeout, pyzx_timeout in [(True, False), (False, True), (True, True)]:
            with self.subTest(db=db_timeout, pyzx=pyzx_timeout), TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'blocks').mkdir(); (root/'inputs').mkdir()
                plan = make_plan('adaptive')
                plan['cases'] = [dict(family='identity', size=2, depth=None, seed=0, scale='patterns', id='test')]
                path = root/'plan.json'; path.write_text(json.dumps(plan))
                graph = build(plan['cases'][0])
                driver = Mock()
                driver.__enter__ = Mock(return_value=driver); driver.__exit__ = Mock(return_value=False)
                session = Mock(); session.run.return_value.single.return_value = {'n': 0}
                session.run.return_value.data.return_value = []
                driver.session.return_value.__enter__ = Mock(return_value=session)
                driver.session.return_value.__exit__ = Mock(return_value=False)
                database = Mock(driver=driver)
                result = (None, dict(status='timeout', reason='Timed out', reduce_s=None)) if pyzx_timeout else (
                    graph, dict(status='ok', setup_s=0.01, reduce_s=0.01))
                with patch('neo4j.GraphDatabase.driver', return_value=driver), \
                     patch('zxdb.zxdb.ZXdb', return_value=database), \
                     patch('evaluation.research_core.load', side_effect=TimeoutError('deadline exceeded') if db_timeout else None) as loader, \
                     patch('evaluation.research_core.read', return_value=graph), \
                     patch('evaluation.research.reduce_database'), \
                     patch('evaluation.research_pyzx.isolated_reference', return_value=result) as pyzx:
                    worker(path, 0, root)
                history = records(root/'blocks'/'test.jsonl')
                both = db_timeout and pyzx_timeout
                self.assertEqual(loader.call_count, 1 if db_timeout or both else 3)
                self.assertEqual(pyzx.call_count, 1 if pyzx_timeout or both else 3)
                self.assertEqual(any(r['kind'] == 'limit_stop' for r in history), both)
                trials = [r for r in history if r['kind'] == 'trial']
                self.assertEqual(len(trials), 1 if both else 3)
                if not both:
                    failed = 'db' if db_timeout else 'pyzx'
                    self.assertEqual(trials[-1]['engines'][failed]['status'], 'not_run_after_timeout')

    def test_hard_timeout_consumes_retry_budget_and_blocks_only_its_series(self):
        import subprocess
        with TemporaryDirectory() as tmp:
            root = Path(tmp); plan = self.plan(); output = root/'results'
            path = root/'plan.json'; path.write_text(json.dumps(plan))
            args = SimpleNamespace(plan=str(path), output=str(output), budget_seconds=1000,
                                   restart_container='memgraph', max_retries=5)
            clock = [0.0]
            def child(command, **kwargs):
                clock[0] += kwargs['timeout']
                raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            with patch('evaluation.research.environment', return_value=dict(source_hashes={})), \
                 patch('evaluation.research.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('evaluation.research.subprocess.run', side_effect=child) as runner, \
                 patch('evaluation.research_recovery.check_container'), \
                 patch('evaluation.research_recovery.ready', return_value=True), \
                 patch('evaluation.research_recovery.recover') as recovery, \
                 patch('evaluation.research_recovery.cleanup_owned'):
                run(args)
                self.assertEqual(runner.call_count, 1)
                self.assertEqual(recovery.call_count, 1)
                run(args)
                self.assertEqual(runner.call_count, 1)
            outcomes = json.loads((output/'campaign.json').read_text())['outcomes']
            self.assertEqual([o['status'] for o in outcomes], ['case_budget_exhausted']+['not_run_case_budget']*4)

    def test_old_large_plan_is_hard_capped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); plan = make_plan(); plan['cases'] = plan['cases'][:1]
            plan['block_timeout_s'] = 7200
            path = root/'plan.json'; path.write_text(json.dumps(plan)); output = root/'results'
            args = SimpleNamespace(plan=str(path), output=str(output), budget_seconds=300)
            with patch('evaluation.research.environment', return_value=dict(source_hashes={})), \
                 patch('evaluation.research.subprocess.run', return_value=SimpleNamespace(returncode=0)) as child:
                run(args)
                self.assertEqual(child.call_args.kwargs['timeout'], 120)


if __name__ == '__main__':
    unittest.main()
