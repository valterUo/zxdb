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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neo4j import GraphDatabase
from evaluate import benchmark_cases
from evaluation.workloads import RULES

ROOT = Path(__file__).resolve().parents[2]


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
    plan = json.loads((output/'plan.json').read_text())
    initial = plan['source_hashes']
    if source_hashes() != initial:
        raise RuntimeError('Measured sources differ from original campaign')
    if (output/'continuation-started.json').exists():
        raise RuntimeError('Continuation already started; inspect before retrying')
    state = json.loads((output/'progress-before-skip-256.json').read_text())
    jobs = plan['jobs']
    interrupted = state['current']
    interrupted_case = output/interrupted['directory']
    handled = {c['index'] for c in state['completed']}
    skipped = []
    now = datetime.now(timezone.utc)
    for index, job in enumerate(jobs):
        if index in handled or job['rule'] != 'full_reduce' or job['spec']['size'] != 256:
            continue
        entry = dict(index=index, **job, status='skipped',
                     reason='User requested skipping remaining 256-qubit full-reduce cases',
                     wall_seconds=0, directory='')
        if index == interrupted['index']:
            entry['directory'] = interrupted['directory']
            entry['wall_seconds'] = (now-datetime.fromisoformat(state['updated'])).total_seconds()
            entry['partial_attempt_preserved'] = True
            write(interrupted_case/'manual-interruption.json', dict(timestamp_utc=now.isoformat(), **entry))
        state['completed'].append(entry)
        skipped.append(entry)
        handled.add(index)
    write(output/'skip-decisions.json', dict(timestamp_utc=now.isoformat(), skipped=skipped,
          policy='Keep all earlier observations; exclude skipped cases from complete performance summaries'))
    write(output/'continuation-started.json', dict(timestamp_utc=now.isoformat(), pid=os.getpid(),
          script=str(Path(__file__).resolve()), source_hashes=initial))
    state.update(status='running', stage='performance', current=None, pid=os.getpid())
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
        recover(interrupted_case)
        performance = output/'performance'
        # Rebuild the aggregate once from preserved canonical observations,
        # including any fully recorded repetitions of the interrupted case.
        with (performance/'observations.jsonl').open('w', encoding='utf-8') as stream:
            for observations in sorted((output/'cases').glob('*/observations.jsonl')):
                stream.write(observations.read_text())
        for index, job in enumerate(jobs):
            if index in handled:
                continue
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
        state.update(status='finished_with_skips', current=None, source_hashes_unchanged=source_hashes() == initial)
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
