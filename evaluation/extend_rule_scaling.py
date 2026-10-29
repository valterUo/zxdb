"""Run larger standalone workloads, with fixed ceilings and preserved failures.

Each case uses the unchanged three-engine timing/verification implementation.
The process deadline also bounds PyZX calls that cannot be interrupted in-process.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from neo4j import GraphDatabase
from evaluation.workloads import RULES


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ['zxdb/main_queries.json', 'zxdb/zxdb.py', 'zxdb_qm/zxqmcpp.cpp',
             'evaluation/core.py', 'evaluation/patterns.py', 'evaluation/workloads.py']


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container', default='zxdb-rule-audit')
    parser.add_argument('--budget', type=int, default=3600, help='Campaign wall seconds; 0 means no shared cutoff')
    parser.add_argument('--case-timeout', type=int, default=1200)
    parser.add_argument('--engine-timeout', type=int, default=300)
    parser.add_argument('--retry-from', type=Path,
                        help='Retry failed or unfinished jobs from an earlier campaign')
    parser.add_argument('--families', nargs='+', help='Restrict jobs to these workload families')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    initial = hashes()
    # Same inputs as before, only the size parameter changes. Dense fans can
    # materialize O(n^2) edges, so their maximum is 1024 rather than 16384.
    dense = {'lcomp', 'pivot', 'pivot_gadget', 'pivot_boundary', 'bialgebra'}
    rules = [r for r in RULES if r != 'full_reduce']
    jobs = []
    for size, copies in ((512, 2048), (1024, 4096), (4096, 8192), (16384, 16384)):
        for rule in rules:
            if rule not in dense or size <= 1024:
                jobs.append(dict(rule=rule, spec=dict(family=rule, size=size, seed=0)))
            jobs.append(dict(rule=rule, spec=dict(family='copies_' + rule, size=copies, seed=0)))
        jobs.append(dict(rule='identity', spec=dict(family='identity_chain', size=copies, seed=0)))
    if args.retry_from:
        previous = json.loads((args.retry_from / 'progress.json').read_text())
        failed = [p for p in previous if p['status'] in ('error', 'case_timeout')]
        previous_plan = json.loads((args.retry_from / 'plan.json').read_text())
        recorded = {(p['rule'], json.dumps(p['spec'], sort_keys=True)) for p in previous}
        failed += [dict(p, status='interrupted') for p in previous_plan['jobs']
                   if (p['rule'], json.dumps(p['spec'], sort_keys=True)) not in recorded]
        failed.sort(key=lambda p: p['status'] != 'case_timeout')
        jobs = [dict(rule=p['rule'], spec=p['spec']) for p in failed]
        if not jobs:
            raise ValueError('No failed or timed-out cases to retry')
    if args.families:
        jobs = [p for p in jobs if p['spec']['family'] in args.families]
        if not jobs:
            raise ValueError('No jobs match the requested families')
    plan = dict(created=datetime.now(timezone.utc).isoformat(), jobs=jobs,
                source_hashes=initial, budget_seconds=args.budget or None,
                case_timeout_seconds=args.case_timeout, repetitions=3, warmups=1,
                engine_timeout_seconds=args.engine_timeout,
                retry_from=str(args.retry_from.resolve()) if args.retry_from else None,
                stop_policy='Stop a workload family after an error or case timeout; never stop based on wins.',
                scope='Larger standalone rules only; no query or graph-generator changes.')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    uri = os.environ.get('ZXDB_URI', 'bolt://127.0.0.1:7691')
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    started = time.monotonic()
    stopped, statuses = set(), []
    for index, job in enumerate(jobs):
        family, size = job['spec']['family'], job['spec']['size']
        case_dir = output / f'{index:03d}-{family}-{size}'
        status = dict(**job, directory=case_dir.name)
        remaining = args.budget - (time.monotonic() - started) if args.budget else float('inf')
        if family in stopped or remaining < 30:
            status['status'] = 'skipped_after_failure' if family in stopped else 'budget_not_run'
        else:
            if hashes() != initial:
                raise RuntimeError('Protected implementation changed during the campaign')
            if args.retry_from:
                # Fresh server for each retry; this does not clear stored data.
                subprocess.run(['docker', 'restart', args.container], check=True, timeout=60,
                               stdout=subprocess.DEVNULL)
                for attempt in range(30):
                    try:
                        with GraphDatabase.driver(uri, auth=auth) as driver:
                            driver.verify_connectivity()
                        break
                    except Exception:
                        if attempt == 29:
                            raise
                        time.sleep(1)
            # The benchmark owns only its UUID-scoped nodes. Never clear an
            # unknown graph, even if a previous timed-out worker left it behind.
            with GraphDatabase.driver(uri, auth=auth) as driver:
                with driver.session() as session:
                    if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                        raise RuntimeError('Dedicated database is not empty; inspect before continuing')
            case_dir.mkdir()
            manifest = case_dir / 'manifest.json'
            manifest.write_text(json.dumps([job]))
            print(f'CASE {index+1}/{len(jobs)} {family} size={size}', flush=True)
            start = time.monotonic()
            with (case_dir / 'run.log').open('w') as log:
                try:
                    process = subprocess.run([sys.executable, '-m', 'evaluation.rerun_three_engines',
                                              '--output', str(case_dir), '--container', args.container,
                                              '--manifest', str(manifest), '--stop-on-error',
                                              '--engine-timeout', str(args.engine_timeout)],
                                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=min(args.case_timeout, remaining))
                    status['status'] = ('complete' if process.returncode == 0 and
                                        (case_dir / 'completion.json').exists() else 'error')
                except subprocess.TimeoutExpired:
                    status['status'] = 'case_timeout'
            status['wall_seconds'] = time.monotonic() - start
            if status['status'] != 'complete':
                stopped.add(family)
                state = subprocess.run(['docker', 'inspect', args.container, '--format', '{{json .State}}'],
                                       capture_output=True, text=True, timeout=30)
                (case_dir / 'container-state-at-failure.json').write_text(state.stdout)
                logs = subprocess.run(['docker', 'logs', '--tail', '50', args.container],
                                      capture_output=True, text=True, timeout=30)
                (case_dir / 'container-log-at-failure.txt').write_text(logs.stdout + logs.stderr)
                # Restart only this dedicated container to cancel any surviving query.
                subprocess.run(['docker', 'restart', args.container], check=True, timeout=60,
                               stdout=subprocess.DEVNULL)
                for attempt in range(30):
                    try:
                        with GraphDatabase.driver(uri, auth=auth) as driver:
                            with driver.session() as session:
                                gidfile = case_dir / 'graph-id.json'
                                if gidfile.exists():
                                    session.run('MATCH(n {graph_id:$gid}) DETACH DELETE n',
                                                gid=json.loads(gidfile.read_text())).consume()
                        break
                    except Exception:
                        if attempt == 29:
                            raise
                        time.sleep(1)
            print(json.dumps(status), flush=True)
        statuses.append(status)
        (output / 'progress.json').write_text(json.dumps(statuses, indent=2))
    (output / 'integrity.json').write_text(json.dumps(dict(
        unchanged=hashes() == initial, before=initial, after=hashes()), indent=2))
    print('Finished. Full status:', output / 'progress.json', flush=True)


if __name__ == '__main__':
    main()
