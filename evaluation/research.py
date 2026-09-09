"""Reproducible paired benchmarks: plan, run, report. See RESEARCH.md."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]


class CaseBudgetExhausted(TimeoutError):
    """Insufficient case time to give another engine its declared allowance."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def reduce_database(db, rule, graph_id, variant='control'):
    from evaluation.research_workloads import RULES
    if variant not in ('control', 'optimized', 'worklist'):
        raise ValueError('Unknown database variant')
    method = RULES[rule][0]
    if variant == 'control':
        return getattr(db, method)(graph_id=graph_id)
    options = dict(exclusive_access=True, incremental=variant == 'worklist')
    if rule == 'full_reduce':
        return db.full_reduce_optimized(graph_id, **options)
    return db.apply_rule_optimized(method, graph_id, **options)


def source_hashes():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder, pattern in [('evaluation', '*.py'), ('zxdb', '*.py'), ('zxdb', '*.json')]
            for p in sorted((ROOT / folder).glob(pattern))}


def environment():
    try:
        commit = subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                          'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), git_commit=commit, source_hashes=source_hashes(),
                packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                          if d.metadata['Name']},
                worker_threads=1, server_cpu_memory_limits='Not measured; record manually in machine-notes.md')


def worker(plan_path, index, directory):
    # Only the child imports numerical libraries, after thread limits are set.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from neo4j import GraphDatabase
    from zxdb.zxdb import ZXdb
    from evaluation.research_workloads import build, digest, payload, metrics, rule_for, RULES, opportunities
    from evaluation.research_core import CountingDriver, load, read, reference, verify, validate_graph, compare_outputs
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if importlib.metadata.version('pyzx') != '0.10.3':
        raise RuntimeError('Baseline contract is audited for PyZX 0.10.3; install that version before benchmarking')
    spec = plan['cases'][index]
    directory = Path(directory)
    path = directory / 'blocks' / (spec['id'] + '.jsonl')
    from evaluation.research_recovery import records
    previous = records(path, repair=True)
    finished_trials = {r['trial'] for r in previous if r.get('kind') == 'trial' and not r['warmup']}
    gid = 'research_' + uuid.uuid4().hex
    uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    db = None
    raw_driver = None
    counter = None
    case_timeout = float(os.environ.get('ZXDB_BLOCK_TIMEOUT_S', plan['block_timeout_s']))
    deadline = time.monotonic() + case_timeout - 5
    def emit(row):
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(case=spec, source_version=os.environ.get('ZXDB_SOURCE_VERSION'), **row), sort_keys=True) + '\n')
            f.flush(); os.fsync(f.fileno())
    try:
        # Check before ZXdb's constructor switches the server storage mode.
        with GraphDatabase.driver(uri, auth=auth) as preflight:
            with preflight.session() as session:
                if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                    raise RuntimeError('Benchmark requires an empty dedicated database; nothing was deleted')
        db = ZXdb(uri=uri, user=auth[0], password=auth[1])
        raw_driver = db.driver
        counter = CountingDriver(raw_driver, plan['query_timeout_s'], deadline)
        db._driver = counter
        with raw_driver.session() as session:
            server = {}
            for name, query in [('version', 'SHOW VERSION'), ('storage', 'SHOW STORAGE INFO'),
                                ('indexes', 'SHOW INDEX INFO')]:
                try:
                    server[name] = session.run(query).data()
                except Exception as exc:
                    server[name] = dict(unavailable=type(exc).__name__)
        emit(dict(kind='environment', server=server, graph_id=gid, case_timeout_s=case_timeout))
        t = time.perf_counter(); original = build(spec); generation_s = time.perf_counter()-t
        validate_graph(original)
        if original.num_vertices() > plan['max_vertices']:
            raise ValueError('Graph exceeds configured max_vertices')
        if original.num_edges() > plan.get('max_edges', float('inf')):
            raise ValueError('Graph exceeds configured max_edges')
        original_payload = payload(original)
        if any(r.get('kind') == 'input' and r.get('sha256') != digest(original_payload) for r in previous):
            raise ValueError('Regenerated input differs from this block checkpoint; use a new campaign')
        write_json(directory / 'inputs' / (spec['id'] + '.json'), original_payload)
        planted = opportunities(spec)
        # Certify the named rule independently, outside all timed trials.
        target_rule = rule_for(spec['family'])
        applicable = None
        if planted:
            # Massive copies keep the same certified neighbourhood; avoid an
            # untimed full-size reduction gating the database measurement.
            probe = (build(dict(spec, family=spec['family'].removeprefix('copies_'), size=2))
                     if plan.get('applicability_probe') == 'motif' else original.copy())
            before_probe = payload(probe)
            reference(probe, target_rule, deadline)
            applicable = payload(probe) != before_probe
            del probe
        emit(dict(kind='input', sha256=digest(original_payload), metrics=metrics(original),
                  generation_s=generation_s, constructed_witnesses=planted,
                  target_rule=target_rule, reference_probe_changed=applicable,
                  applicability_probe=plan.get('applicability_probe', 'full'),
                  baseline_contract='pyzx-0.10.3-native-v2'))
        if planted and not applicable:
            raise ValueError('Constructed workload has no observed target-rule application')
        rule = rule_for(spec['family'])
        repetitions, warmups = plan['repetitions'], plan['warmups']
        rng = random.Random(plan['order_seed'] + index)
        first = rng.randrange(2)
        timed_out_engines = {r['engine'] for r in previous
                             if r.get('kind') == 'engine_limit' and r.get('status') == 'timeout'}
        for trial in range(-warmups, repetitions):
            if trial >= 0 and trial in finished_trials:
                continue
            order = ['db', 'pyzx'] if (trial + first) % 2 == 0 else ['pyzx', 'db']
            engines = {}
            outputs = {}
            for engine in order:
                emit(dict(kind='stage', trial=trial, engine=engine))
                if engine in timed_out_engines:
                    engines[engine] = dict(status='not_run_after_timeout', reduce_s=None,
                                          verification=dict(status='inconclusive', reason='Earlier attempt timed out'))
                    continue
                resource_info = {}
                if plan.get('adaptive'):
                    budget = plan['engine_timeout_s'] if engine == 'db' else plan['pyzx_timeout_s']
                    if deadline-time.monotonic() < budget+15:
                        raise CaseBudgetExhausted('Case budget insufficient for a full engine attempt')
                if engine == 'db':
                    counter.deadline = min(deadline, time.monotonic()+plan.get('engine_timeout_s', float('inf')))
                    try:
                        t = time.perf_counter(); load(db, original, gid); setup = time.perf_counter()-t
                        before_count = counter.submissions
                        t = time.perf_counter(); reduce_database(db, rule, gid, plan.get('db_variant', 'control'))
                        elapsed = time.perf_counter()-t
                        submissions = counter.submissions-before_count
                        t = time.perf_counter(); result = read(db, gid); read_s = time.perf_counter()-t
                    except Exception as exc:
                        from evaluation.research_adaptive import is_timeout
                        if not plan.get('adaptive') or not is_timeout(exc):
                            raise
                        timed_out_engines.add('db')
                        engines[engine] = dict(status='timeout', reduce_s=None,
                                              verification=dict(status='inconclusive', reason=str(exc)))
                        emit(dict(kind='engine_limit', trial=trial, engine='db', status='timeout'))
                        continue
                    finally:
                        counter.deadline = deadline
                else:
                    if plan.get('pyzx_timeout_s'):
                        from evaluation.research_pyzx import isolated_reference
                        result, resource_info = isolated_reference(original, rule,
                            (plan['pyzx_timeout_s'] if plan.get('adaptive') else
                             min(plan['pyzx_timeout_s'], max(.01, deadline-time.monotonic()-15))),
                            plan['pyzx_max_rss_mb'])
                        if result is None:
                            engines[engine] = dict(resource_info, verification=dict(
                                status='inconclusive', reason=resource_info['reason']))
                            if plan.get('adaptive') and resource_info['status'] == 'timeout':
                                timed_out_engines.add('pyzx')
                                emit(dict(kind='engine_limit', trial=trial, engine='pyzx', status='timeout'))
                            continue
                        setup, elapsed = resource_info['setup_s'], resource_info['reduce_s']
                    else:
                        t = time.perf_counter(); result = original.copy(); setup = time.perf_counter()-t
                        t = time.perf_counter(); reference(result, rule, deadline); elapsed = time.perf_counter()-t
                    submissions, read_s = 0, 0
                t = time.perf_counter(); check = verify(original, result); check_s = time.perf_counter()-t
                outputs[engine] = result
                # Vertex labels differ on DB read. Compare layout-free structural
                # statistics for progress; the reference probe above uses exact topology.
                changed = metrics(original) != metrics(result)
                engines[engine] = dict(reduce_s=elapsed, setup_s=setup, read_s=read_s,
                                       validation_s=check_s, verification=check,
                                       output=metrics(result), output_sha256=digest(payload(result)),
                                       metrics_changed=changed, cypher_submissions=submissions)
                engines[engine].update(resource_info)
                if engine == 'db':
                    engines[engine]['variant'] = plan.get('db_variant', 'control')
                if trial == 0 or check['status'] == 'failed':
                    write_json(directory / 'inputs' / f'{spec["id"]}-{engine}-output.json', payload(result))
            comparison = (compare_outputs(outputs['db'], outputs['pyzx']) if len(outputs) == 2 else
                          {level: dict(status='inconclusive', reason='Engine resource limit; output unavailable')
                           for level in ('tensor', 'graph', 'degree_sequence')})
            if comparison['tensor']['status'] == 'failed':
                for engine, result in outputs.items():
                    write_json(directory / 'inputs' / f'{spec["id"]}-{trial}-{engine}-mismatch.json', payload(result))
            no_progress = planted and 'db' in outputs and not engines['db']['metrics_changed']
            emit(dict(kind='trial', trial=trial, warmup=trial < 0, order=order, engines=engines,
                      comparison=comparison,
                      status='failed' if comparison['tensor']['status'] == 'failed' else 'no_progress' if no_progress else 'failed' if any(
                          e['verification']['status'] == 'failed' for e in engines.values()) else
                          'resource_limit' if len(outputs) < 2 else 'ok'))
            if timed_out_engines == {'db', 'pyzx'}:
                emit(dict(kind='limit_stop', reason='both_engines_timeout'))
                break
        emit(dict(kind='complete'))
    except Exception as exc:
        if isinstance(exc, CaseBudgetExhausted):
            emit(dict(kind='case_budget', reason=str(exc)))
        emit(dict(kind='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                  last_query=counter.last_query if counter else None,
                  cypher_submissions=counter.submissions if counter else 0))
        raise
    finally:
        if db is not None:
            primary_error = sys.exc_info()[0] is not None
            # Bypass an expired local deadline for cleanup; still bound the query.
            from neo4j import Query
            try:
                with raw_driver.session() as session:
                    session.run(Query('MATCH(n:Node {graph_id:$gid}) DETACH DELETE n', timeout=10), gid=gid).consume()
            except Exception as exc:
                emit(dict(kind='cleanup_error', error=f'{type(exc).__name__}: {exc}', graph_id=gid))
                if not primary_error:
                    raise
            finally:
                db.close()


def run(args):
    from evaluation.research_recovery import campaign_lock
    with campaign_lock(args.output):
        return _run_campaign(args)


def _run_campaign(args):
    directory = Path(args.output).resolve()
    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise ValueError(f'Plan file does not exist: {plan_path}. Use the plan generated before the first run (for example study-plan.json), not a file inside a new results folder.')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    case_cap = min(plan['block_timeout_s'], getattr(args, 'case_timeout_seconds', 120))
    if case_cap <= 10:
        raise ValueError('Case timeout must exceed 10 seconds')
    if plan.get('adaptive'):
        policy = plan['adaptive']
        if policy.get('consecutive_timeouts', 0) < 1 or policy.get('stop_on') not in ('both', 'either'):
            raise ValueError('Invalid adaptive stopping policy')
    if plan.get('schema') != 1 or plan['repetitions'] < 2 or plan['warmups'] < 1:
        raise ValueError('Expected schema 1, at least two repetitions and one warmup')
    if plan['block_timeout_s'] <= 10 or plan['query_timeout_s'] <= 0:
        raise ValueError('Invalid timeout settings')
    if plan.get('db_variant', 'control') not in ('control', 'optimized', 'worklist'):
        raise ValueError('Unknown database variant')
    if plan.get('baseline_contract', 'pyzx-0.10.3-native-v2') != 'pyzx-0.10.3-native-v2':
        raise ValueError('Unsupported baseline contract')
    if plan.get('pyzx_timeout_s') is not None:
        if plan['pyzx_timeout_s'] <= 0 or plan.get('pyzx_max_rss_mb', 0) <= 0:
            raise ValueError('PyZX isolation requires positive time and RSS limits')
        importlib.metadata.version('psutil')  # Fail before starting a campaign if unavailable.
    ids = [c['id'] for c in plan['cases']]
    if len(ids) != len(set(ids)) or any(not i.isalnum() for i in ids):
        raise ValueError('Case IDs must be unique and alphanumeric')
    # One command starts new campaigns or continues recognized checkpoints.
    # --resume is retained as a backward-compatible, optional flag.
    if directory.exists() and not directory.is_dir():
        raise ValueError('Output path is not a directory')
    resume = (directory / 'plan.json').is_file()
    container = getattr(args, 'restart_container', None)
    max_retries = getattr(args, 'max_retries', 1)
    if max_retries < 0:
        raise ValueError('--max-retries must be nonnegative')
    if directory.exists() and not resume and any(directory.iterdir()):
        raise ValueError('Output directory contains files but no campaign plan; refusing to overwrite it')
    current_environment = environment()
    version = hashlib.sha256(json.dumps(dict(source=current_environment['source_hashes'],
                                             packages=current_environment.get('packages', {})), sort_keys=True).encode()).hexdigest()
    from evaluation.research_recovery import records, append, ready, recover, cleanup_owned, check_container
    if resume:
        if not directory.exists() or json.loads((directory / 'plan.json').read_text()) != plan:
            raise ValueError('Resume requires the existing directory and exactly the same saved plan')
        saved_environment = json.loads((directory / 'environment.json').read_text())
        sessions = records(directory / 'sessions.jsonl', repair=True)
        if sessions:
            saved_environment = sessions[-1]['environment']
            if plan.get('adaptive') and sessions[-1].get('case_timeout_s', case_cap) != case_cap:
                raise ValueError('Use a new output directory when changing adaptive time limits')
        if (saved_environment['source_hashes'] != current_environment['source_hashes'] or
            saved_environment.get('packages', {}) != current_environment.get('packages', {})) and (
                plan.get('adaptive') or not getattr(args, 'allow_code_change', False)):
            raise ValueError('Source code changed. ' + ('Adaptive campaigns require a new output directory.'
                             if plan.get('adaptive') else 'Use --allow-code-change to record a new version; mixed-version cells will not be aggregated.'))
    else:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'blocks').mkdir(); (directory / 'inputs').mkdir()
        write_json(directory / 'plan.json', plan)
        write_json(directory / 'environment.json', current_environment)
    append(directory / 'sessions.jsonl', dict(time=time.time(), resume=resume, source_version=version,
                                              environment=current_environment, restart_container=container,
                                              max_retries=max_retries, case_timeout_s=case_cap))
    uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    if container:
        check_container(container, uri)
        if not resume and not ready(uri, auth):
            recover(container, uri, auth, directory, 'initial database unavailable')
    if resume:
        for path in (directory / 'blocks').glob('*.jsonl'):
            records(path, repair=True)
        if not ready(uri, auth):
            if not container:
                raise RuntimeError('Memgraph is unavailable. Supply --restart-container to enable recovery')
            recover(container, uri, auth, directory, 'resume with unavailable database')
        else:
            cleanup_owned(uri, auth, directory)
    from evaluation.research_adaptive import ordered_cases, stop_evidence, blocked_predecessor
    order = ordered_cases(plan)
    start = time.monotonic()
    outcomes = []
    child_env = dict(os.environ, ZXDB_SOURCE_VERSION=version)
    histories = {s['id']: records(directory / 'blocks' / (s['id'] + '.jsonl')) for s in plan['cases']}
    def pending_outcomes(indices, default):
        result = []
        for i in indices:
            cid = plan['cases'][i]['id']
            history = records(directory / 'blocks' / (cid + '.jsonl'))
            completed = {r['trial'] for r in history if r.get('kind') == 'trial' and not r['warmup']}
            failures = sum(r.get('kind') in ('error', 'timeout') for r in history)
            status = ('complete' if completed == set(range(plan['repetitions'])) else
                      'both_engines_timeout' if any(r.get('kind') == 'limit_stop' for r in history) else
                      'case_budget_exhausted' if any(r.get('kind') == 'case_budget' for r in history) else
                      'not_run_case_budget' if any(r.get('kind') == 'adaptive_skip' and
                                                  r.get('reason') == 'predecessor_case_budget' for r in history) else
                      'not_run_adaptive_stop' if any(r.get('kind') == 'adaptive_skip' for r in history) else
                      'failed_retries_exhausted' if failures > max_retries else default)
            result.append(dict(id=cid, status=status))
        return result
    for position, index in enumerate(order):
        spec = plan['cases'][index]
        path = directory / 'blocks' / (spec['id'] + '.jsonl')
        prior = records(path)
        if any(r.get('kind') == 'limit_stop' for r in prior):
            outcomes.append(dict(id=spec['id'], status='both_engines_timeout'))
            continue
        if any(r.get('kind') == 'case_budget' for r in prior):
            outcomes.append(dict(id=spec['id'], status='case_budget_exhausted'))
            continue
        evidence = stop_evidence(plan, spec, histories)
        blocked = blocked_predecessor(plan, spec, histories)
        if evidence or blocked:
            if not any(r.get('kind') == 'adaptive_skip' for r in prior):
                append(path, dict(case=spec, kind='adaptive_skip',
                                  reason='consecutive_both_engine_timeouts' if evidence else 'predecessor_case_budget',
                                  evidence=evidence or blocked, source_version=version))
            outcomes.append(dict(id=spec['id'], status='not_run_adaptive_stop' if evidence else 'not_run_case_budget',
                                 evidence=evidence or blocked))
            continue
        done = {r['trial'] for r in prior if r.get('kind') == 'trial' and not r['warmup']}
        if done == set(range(plan['repetitions'])):
            outcomes.append(dict(id=spec['id'], status='complete'))
            continue
        failures = sum(r.get('kind') in ('error', 'timeout') for r in prior)
        if failures > max_retries:
            outcomes.append(dict(id=spec['id'], status='failed_retries_exhausted'))
            continue
        remaining = args.budget_seconds - (time.monotonic()-start)
        if remaining < case_cap:
            outcomes.extend(pending_outcomes(order[position:], 'not_run_budget'))
            break
        print(f'{position+1}/{len(order)} {spec}', flush=True)
        stop = False
        spent = sum(r.get('worker_wall_s', 0) for r in prior if r.get('kind') == 'attempt')
        while True:
            attempt_cap = min(case_cap-spent, args.budget_seconds-(time.monotonic()-start))
            if attempt_cap <= 10:
                status = 'case_budget_exhausted'
                if not any(r.get('kind') == 'case_budget' for r in records(path)):
                    append(path, dict(case=spec, kind='case_budget', source_version=version,
                                      reason='Hard case budget exhausted; both-engine performance unknown'))
                break
            status = 'error'
            attempt_start = time.monotonic()
            try:
                with (directory / 'blocks' / (spec['id'] + '.log')).open('a', encoding='utf-8') as log:
                    p = subprocess.run([sys.executable, '-m', 'evaluation.research', '_worker',
                                        '--plan', str(directory / 'plan.json'), '--index', str(index),
                                        '--output', str(directory)], cwd=ROOT,
                                       env=dict(child_env, ZXDB_BLOCK_TIMEOUT_S=str(attempt_cap)),
                                       stdout=log, stderr=subprocess.STDOUT, timeout=attempt_cap)
                if p.returncode == 0:
                    status = ('both_engines_timeout' if any(r.get('kind') == 'limit_stop' for r in records(path))
                              else 'complete')
                    break
            except subprocess.TimeoutExpired:
                status = 'timeout'
                records(path, repair=True)
                append(path, dict(case=spec, kind='timeout', error='Worker wall-clock timeout',
                                  timeout_s=attempt_cap, source_version=version))
            finally:
                elapsed = time.monotonic()-attempt_start
                spent += elapsed
                records(path, repair=True)
                append(path, dict(case=spec, kind='attempt', worker_wall_s=elapsed, source_version=version))
            updated = records(path, repair=True)
            new_failures = sum(r.get('kind') in ('error', 'timeout') for r in updated)
            if new_failures == failures:
                append(path, dict(case=spec, kind='error', error='Worker exited without an error record', source_version=version))
                new_failures += 1
            failures = new_failures
            if not container:
                stop = True
                break
            # Restart even when the next case will be skipped: kill any query
            # still running after a worker timeout and restore a clean server.
            try:
                if status == 'timeout' or not ready(uri, auth):
                    recover(container, uri, auth, directory, f'{spec["id"]}: {status}')
                else:
                    cleanup_owned(uri, auth, directory)
            except Exception as exc:
                append(directory / 'recovery.jsonl', dict(event='campaign_stopped', error=str(exc), time=time.time()))
                status = 'recovery_failed'; stop = True
                break
            if any(r.get('kind') == 'limit_stop' for r in updated):
                status = 'both_engines_timeout'
                break
            if spent >= case_cap-10 or any(r.get('kind') == 'case_budget' for r in updated):
                if not any(r.get('kind') == 'case_budget' for r in updated):
                    append(path, dict(case=spec, kind='case_budget', source_version=version,
                                      reason='Hard case budget exhausted; both-engine performance unknown'))
                status = 'case_budget_exhausted'
                break
            if failures > max_retries:
                status = 'failed_retries_exhausted'
                break
            if args.budget_seconds - (time.monotonic()-start) < case_cap:
                status = 'not_run_budget'; stop = True
                break
            print(f'Recovered Memgraph; continuing {spec["id"]} from saved trials (retry {failures}/{max_retries})', flush=True)
        outcomes.append(dict(id=spec['id'], status=status))
        histories[spec['id']] = records(path)
        if stop:
            outcomes.extend(pending_outcomes(order[position+1:], 'not_run_after_error'))
        write_json(directory / 'campaign.json', dict(outcomes=outcomes, elapsed_s=time.monotonic()-start))
        if stop:
            break
    write_json(directory / 'campaign.json', dict(outcomes=outcomes, elapsed_s=time.monotonic()-start))
    from evaluation.research_report import report
    report(directory)
    if any(o['status'] in ('error', 'timeout', 'recovery_failed') for o in outcomes):
        raise RuntimeError(f'Campaign stopped after a worker failure; inspect {directory / "campaign.json"}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--profile', choices=['smoke', 'study', 'massive', 'adaptive'], default='smoke')
    p.add_argument('--db-variant', choices=['control', 'optimized', 'worklist'], default='control')
    p.add_argument('--output', required=True)
    p.add_argument('--families', nargs='+', help='Optional subset of workload families')
    p.add_argument('--sizes', nargs='+', type=int, help='Override scale grid (use with --families)')
    p = sub.add_parser('run'); p.add_argument('--plan', required=True); p.add_argument('--output', required=True)
    p.add_argument('--budget-seconds', type=float, default=3600)
    p.add_argument('--case-timeout-seconds', type=float, default=120,
                   help='Maximum worker seconds per case across retries (default: 120)')
    p.add_argument('--resume', action='store_true', help='Optional compatibility flag; run automatically starts or resumes')
    p.add_argument('--restart-container', help='Opt in to restarting this dedicated local Memgraph Docker container')
    p.add_argument('--max-retries', type=int, default=1, help='Retries per case across resumed sessions; exhausted cases are skipped')
    p.add_argument('--allow-code-change', action='store_true', help='Record changed source versions on resume; do not aggregate mixed-version cells')
    p = sub.add_parser('report'); p.add_argument('directory')
    p = sub.add_parser('_worker'); p.add_argument('--plan', required=True); p.add_argument('--index', type=int, required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        from evaluation.research_workloads import FAMILIES, digest, make_plan
        target = Path(args.output)
        if target.exists():
            raise ValueError('Plan already exists; edit it explicitly or choose a new filename')
        plan = make_plan(args.profile)
        plan['db_variant'] = args.db_variant
        if args.families:
            if set(args.families) - set(FAMILIES):
                raise ValueError(f'Unknown families; choose from {FAMILIES}')
            plan['cases'] = [s for s in plan['cases'] if s['family'] in args.families]
        if args.sizes:
            if not args.families or min(args.sizes) < 2:
                raise ValueError('--sizes requires --families and sizes >= 2')
            bases = {(s['family'], s['depth'], s['seed']): s for s in plan['cases']}
            plan['cases'] = [dict(s, size=size) for s in bases.values() for size in sorted(set(args.sizes))]
            for spec in plan['cases']:
                spec.pop('id', None); spec['id'] = digest(spec)[:16]
        write_json(target, plan)
    elif args.command == 'run':
        run(args)
    elif args.command == 'report':
        from evaluation.research_report import report
        report(Path(args.directory))
    else:
        worker(args.plan, args.index, args.output)


if __name__ == '__main__':
    main()
