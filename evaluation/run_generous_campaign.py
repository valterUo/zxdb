"""Serial, isolated full rerun with durable progress and no shared time cutoff."""
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
from evaluate import benchmark_cases
from evaluation.workloads import RULES

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    paths = [ROOT/'evaluate.py', *sorted((ROOT/'evaluation').glob('*.py')),
             *sorted((ROOT/'zxdb').glob('*.py')), *sorted((ROOT/'tests').glob('*.py')),
             ROOT/'zxdb/main_queries.json', ROOT/'zxdb_qm/zxqmcpp.cpp']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise RuntimeError('Choose a fresh campaign directory')
    initial = source_hashes()
    jobs = [dict(rule=r, spec=s) for r, s, _ in benchmark_cases(
        False, qubits=[3, 8, 16, 32, 64, 128, 256], seeds=3, depth_factor=20)]
    jobs += [dict(rule='identity', spec=dict(family='identity_chain', size=n, seed=0))
             for n in (2, 16, 128, 512)]
    dense = {'lcomp', 'pivot', 'pivot_gadget', 'pivot_boundary', 'bialgebra'}
    for size, copies in ((512, 2048), (1024, 4096), (4096, 8192), (16384, 16384)):
        for rule in RULES:
            if rule == 'full_reduce':
                continue
            if rule not in dense or size <= 1024:
                jobs.append(dict(rule=rule, spec=dict(family=rule, size=size, seed=0)))
            jobs.append(dict(rule=rule, spec=dict(family='copies_'+rule, size=copies, seed=0)))
        jobs.append(dict(rule='identity', spec=dict(family='identity_chain', size=copies, seed=0)))
    # Complete the circuit grid before the potentially multi-day largest rule cases.
    jobs.sort(key=lambda j: (j['rule'] != 'full_reduce', j['spec']['size'], j['spec']['family'], j['spec']['seed']))
    plan = dict(created=datetime.now(timezone.utc).isoformat(), jobs=jobs,
                source_hashes=initial, repetitions=3, warmups=1,
                engine_timeout_seconds=7200, case_timeout_seconds=93600,
                campaign_timeout_seconds=None, container=args.container,
                uri=os.environ['ZXDB_URI'],
                timeout_policy='Database watchdog and cooperative PyZX deadline: 2h. Hard case process deadline: 26h, including checks. No family skipping after failure.',
                scope='Fresh full original grid plus extended rules; 256-qubit circuit scale added. Unchanged rewrite implementations and generators.')
    write(output/'plan.json', plan)
    if args.plan_only:
        print(f'{len(jobs)} configurations; {len(jobs)*12} engine trials including warmups')
        return
    state = dict(status='running', stage='correctness', completed=[], current=None, pid=os.getpid())
    def progress():
        state['updated'] = datetime.now(timezone.utc).isoformat()
        write(output/'progress.json', state)
    def run(command, log, timeout):
        with log.open('w', encoding='utf-8') as stream:
            try:
                return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                return -999
    def recover(case):
        subprocess.run(['docker', 'restart', args.container], check=True, timeout=120)
        for attempt in range(60):
            try:
                with GraphDatabase.driver(os.environ['ZXDB_URI'], auth=('', '')) as driver:
                    with driver.session() as session:
                        gidfile = case/'graph-id.json'
                        if gidfile.exists():
                            session.run('MATCH(n {graph_id:$gid}) DETACH DELETE n',
                                        gid=json.loads(gidfile.read_text())).consume()
                        if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                            raise RuntimeError('Unowned data found; stopping without deleting it')
                return
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(1)
    progress()
    try:
        code = run([sys.executable, 'evaluate.py', '--correctness-only', '--variants', 'optimized', 'native',
                    '--regression-timeout', '7200', '--trial-timeout', '7200',
                    '--restart-container', args.container, '--output', str(output/'correctness')],
                   output/'correctness.log', 93600)
        state['correctness_returncode'] = code
        if code:
            raise RuntimeError(f'Correctness gate failed ({code}); inspect correctness evidence before timing')
        performance = output/'performance'
        performance.mkdir()
        state['stage'] = 'performance'
        for index, job in enumerate(jobs):
            if source_hashes() != initial:
                raise RuntimeError('Sources changed during campaign; preserving results and stopping')
            case = output/'cases'/f"{index:03d}-{job['spec']['family']}-{job['spec']['size']}-{job['spec']['seed']}"
            case.mkdir(parents=True)
            write(case/'manifest.json', [job])
            state['current'] = dict(index=index, **job, directory=str(case.relative_to(output)))
            progress()
            print(f'CASE {index+1}/{len(jobs)} {job}', flush=True)
            started = time.monotonic()
            code = run([sys.executable, '-m', 'evaluation.rerun_three_engines', '--output', str(case),
                        '--container', args.container, '--manifest', str(case/'manifest.json'),
                        '--engine-timeout', '7200', '--stop-on-error'], case/'run.log', 93600)
            complete = code == 0 and (case/'completion.json').exists()
            state['completed'].append(dict(**state['current'], status='complete' if complete else 'error',
                                           returncode=code, wall_seconds=time.monotonic()-started))
            progress()
            if not complete:
                for command, filename in [(['docker', 'inspect', args.container], 'container.json'),
                                          (['docker', 'logs', '--tail', '100', args.container], 'container.log')]:
                    run(command, case/filename, 60)
                recover(case)
            if (case/'environment.json').exists():
                env = json.loads((case/'environment.json').read_text())
                env['manifest'] = jobs
                write(performance/'environment.json', env)
            if (case/'observations.jsonl').exists():
                with (performance/'observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write((case/'observations.jsonl').read_text())
            if (performance/'observations.jsonl').exists():
                if run([sys.executable, '-m', 'evaluation.report_three_engines', str(performance)],
                       output/'report.log', 300):
                    raise RuntimeError('Report generation failed')
        state.update(status='finished', current=None, source_hashes_unchanged=source_hashes() == initial)
        if all(c['status'] == 'complete' for c in state['completed']):
            write(performance/'completion.json', dict(finished=True, configurations=len(jobs)))
        run([sys.executable, '-m', 'evaluation.report_three_engines', str(performance)], output/'report.log', 300)
        run([sys.executable, 'evaluation/plot_timings.py', str(performance/'timings.csv'),
             '--output', str(performance/'figures')], output/'plots.log', 300)
        progress()
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        progress()
        raise


if __name__ == '__main__':
    main()
