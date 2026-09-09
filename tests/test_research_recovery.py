import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import os
import subprocess
from unittest.mock import patch

from evaluation.research import run
from evaluation.research_workloads import make_plan
from evaluation.research_recovery import append, records, campaign_lock, check_container


class RecoveryTests(unittest.TestCase):
    def test_partial_record_repaired_without_losing_completed_trials(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'block.jsonl'
            p.write_bytes(b'{"trial":0}\n{"trial":')
            self.assertEqual(records(p, repair=True), [{'trial': 0}])
            append(p, {'trial': 1})
            self.assertEqual(records(p), [{'trial': 0}, {'trial': 1}])
            self.assertEqual(len(list(Path(d).glob('*.partial-*'))), 1)
            p.write_bytes(b'broken\n{"trial":1}\n')
            with self.assertRaises(ValueError):
                records(p, repair=True)

    def test_directory_lock_is_exclusive_and_released(self):
        with TemporaryDirectory() as d:
            with campaign_lock(Path(d) / 'run'):
                with self.assertRaises(RuntimeError):
                    with campaign_lock(Path(d) / 'run'):
                        pass
            with campaign_lock(Path(d) / 'run'):
                pass

    def test_restart_container_must_publish_target_port(self):
        with patch('subprocess.check_output', return_value='{"7687/tcp":[{"HostPort":"7688"}]}'):
            check_container('test', 'bolt://localhost:7688')
            with self.assertRaises(ValueError):
                check_container('test', 'bolt://localhost:7687')
            with self.assertRaises(ValueError):
                check_container('test', 'bolt://example.com:7688')

    def simulate_campaign(self, retries, empty_directory=False):
        with TemporaryDirectory() as d:
            root = Path(d); plan = make_plan(); plan['cases'] = plan['cases'][:2]
            plan_path = root / 'plan.json'; plan_path.write_text(json.dumps(plan))
            output = root / 'results'
            if empty_directory:
                output.mkdir()
            args = SimpleNamespace(plan=str(plan_path), output=str(output), budget_seconds=1000,
                                   resume=True, restart_container='memgraph', max_retries=retries,
                                   allow_code_change=False)
            calls = []
            def child(command, **kwargs):
                index = int(command[command.index('--index')+1]); spec = plan['cases'][index]
                path = output / 'blocks' / (spec['id'] + '.jsonl')
                prior = records(path)
                calls.append(index)
                done = {r['trial'] for r in prior if r.get('kind') == 'trial'}
                def trial(n):
                    append(path, dict(case=spec, kind='trial', trial=n, warmup=False, status='ok',
                                      engines={e: dict(reduce_s=t, verification=dict(status='verified'))
                                               for e, t in [('db', 1), ('pyzx', 2)]}))
                if len(calls) == 1:
                    trial(0)
                    append(path, dict(case=spec, kind='error', error='Simulated server disconnect'))
                    return SimpleNamespace(returncode=1)
                for n in range(plan['repetitions']):
                    if n not in done:
                        trial(n)
                append(path, dict(case=spec, kind='complete'))
                return SimpleNamespace(returncode=0)
            with patch('evaluation.research.environment', return_value={'source_hashes': {'test': 'v1'}}), \
                 patch('evaluation.research.subprocess.run', side_effect=child), \
                 patch('evaluation.research_recovery.check_container'), \
                 patch('evaluation.research_recovery.ready', side_effect=[True] + [False]*10), \
                 patch('evaluation.research_recovery.recover') as restart, \
                 patch('evaluation.research_recovery.cleanup_owned'):
                run(args)
                self.assertEqual(restart.call_count, 1)
                self.assertEqual(len(calls), 3 if retries else 2)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(sum(r['failed_attempts'] for r in summary), 1)
                self.assertTrue(any(r['speedup'] is None for r in summary))
                for p in (output / 'blocks').glob('*.jsonl'):
                    trials = [r['trial'] for r in records(p) if r['kind'] == 'trial']
                    self.assertEqual(len(trials), len(set(trials)))
                # Completed / permanently exhausted blocks must not rerun.
                args.resume = False
                previous_calls = len(calls)
                run(args)
                self.assertEqual(len(calls), previous_calls)
                # Changed source and changed plan cannot silently mix results.
                sessions_path = output / 'sessions.jsonl'
                sessions = records(sessions_path)
                sessions[-1]['environment']['source_hashes'] = {'test': 'v0'}
                sessions_path.write_text(''.join(json.dumps(s)+'\n' for s in sessions))
                with self.assertRaisesRegex(ValueError, 'Source code changed'):
                    run(args)
                args.allow_code_change = True
                run(args)
                plan['repetitions'] += 1; plan_path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, 'same saved plan'):
                    run(args)

    def test_retry_keeps_completed_trials_then_resume_skips_completed_blocks(self):
        self.simulate_campaign(retries=1)

    def test_exhausted_case_is_recorded_and_later_cases_continue(self):
        self.simulate_campaign(retries=0, empty_directory=True)

    def test_unrecognized_output_and_missing_plan_are_not_overwritten(self):
        with TemporaryDirectory() as d:
            root = Path(d); output = root / 'results'; output.mkdir()
            unrelated = output / 'notes.txt'; unrelated.write_text('keep')
            plan = root / 'study-plan.json'; plan.write_text(json.dumps(make_plan()))
            args = SimpleNamespace(plan=str(plan), output=str(output), budget_seconds=10)
            with self.assertRaisesRegex(ValueError, 'no campaign plan'):
                run(args)
            self.assertEqual(unrelated.read_text(), 'keep')
            args.plan = str(output / 'plan.json')
            with self.assertRaisesRegex(ValueError, 'Plan file does not exist'):
                run(args)


@unittest.skipUnless(os.environ.get('ZXDB_RECOVERY_CONTAINER', '').startswith('zxdb-recovery-test'),
                     'Requires an explicitly selected disposable recovery-test container')
class DockerRecoveryTest(unittest.TestCase):
    def test_resume_after_real_container_outage(self):
        with TemporaryDirectory() as d:
            root = Path(d); plan = make_plan(); plan['cases'] = plan['cases'][:1]
            path = root / 'plan.json'; path.write_text(json.dumps(plan))
            output = root / 'results'
            args = SimpleNamespace(plan=str(path), output=str(output), budget_seconds=400,
                                   resume=False, restart_container=os.environ['ZXDB_RECOVERY_CONTAINER'],
                                   max_retries=1, allow_code_change=False)
            run(args)
            block = output / 'blocks' / (plan['cases'][0]['id'] + '.jsonl')
            saved = records(block)
            first = next(r for r in saved if r['kind'] == 'trial' and r['trial'] == 0)
            # Simulate interruption after a complete pair, then a real DB outage.
            retained = [r for r in saved if r['kind'] != 'complete' and not
                        (r['kind'] == 'trial' and r['trial'] == 1)]
            block.write_text(''.join(json.dumps(r)+'\n' for r in retained))
            append(block, dict(case=plan['cases'][0], kind='error', error='Injected outage'))
            subprocess.run(['docker', 'stop', args.restart_container], check=True, timeout=30)
            args.resume = True
            run(args)
            trials = [r for r in records(block) if r['kind'] == 'trial' and not r['warmup']]
            self.assertEqual([r['trial'] for r in trials], [0, 1])
            self.assertEqual(trials[0], first)
            self.assertTrue(any(r['event'] == 'restart_ready' for r in records(output / 'recovery.jsonl')))
            summary = json.loads((output / 'summary.json').read_text())[0]
            self.assertGreater(summary['failed_attempts'], 0)
            self.assertIsNone(summary['speedup'])


if __name__ == '__main__':
    unittest.main()
