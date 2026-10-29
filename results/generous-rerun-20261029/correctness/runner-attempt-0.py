"""Run the complete audit and paired PyZX benchmark: python evaluate.py."""
import argparse
from collections import Counter
from contextlib import redirect_stdout, contextmanager, nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

import pyzx as zx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from evaluation.cases import RULES as CASES
from evaluation.circuits import build_suite
from evaluation.core import CountingDriver, load, read, reference, verify, certify_unitary_rewrite
from evaluation.workloads import RULES, build, digest, metrics, payload
from evaluation.evidence import save_graph, check_reduction_fixed_point
from zxdb.zxdb import ZXdb

ROOT = Path(__file__).resolve().parent


def reduce_database(db, rule, gid, variant):
    method = RULES[rule][0]
    if variant == 'control' or rule == 'hadamard_cancel':
        return getattr(db, method)(gid)
    if rule == 'full_reduce':
        if variant == 'native':
            return db.full_reduce_native(gid)
        return db.full_reduce_optimized(gid, exclusive_access=True)
    if variant == 'native':
        with db._optimized_scope(gid, exclusive_access=True):
            db._reduction_options.get()['native'] = True
            return getattr(db, method)(gid)
    return db.apply_rule_optimized(method, gid, exclusive_access=True)


def zero_cases():
    """Previously hidden audit failures; these must remain visible."""
    for explicit in (False, True):
        g = build(dict(family='identity', size=1, seed=0))
        if explicit:
            g.scalar.is_zero = True
        else:
            g.add_vertex(zx.VertexType.Z, phase=1)
        yield 'explicit_zero_scalar' if explicit else 'isolated_zero_spider', g


def correctness_cases():
    for rule, spec in CASES.items():
        for name, builder, *_ in spec['cases']:
            yield rule, name, builder()
    for name, builder in build_suite():
        yield 'full_reduce', name, builder()
    for name, g in zero_cases():
        yield 'full_reduce', name, g


def benchmark_cases(quick, full_reduce_only=False, qubits=None, seeds=None, depth_factor=10):
    """Predeclared scales: independent matches, fan growth, realistic circuits."""
    for rule in RULES:
        if rule == 'full_reduce' or full_reduce_only:
            continue
        for n in ([2, 8] if quick else [2, 32, 512]):
            spec = dict(family='copies_' + rule, size=n, seed=0)
            yield rule, spec, build(spec)
        for n in ([2] if quick else [2, 16, 128]):
            spec = dict(family=rule, size=n, seed=0)
            yield rule, spec, build(spec)
    for family in ('random_clifford', 'random_clifford_t', 'random_t_heavy',
                   'layered_local', 'layered_global'):
        for q in (qubits or ([3] if quick else [3, 8, 16])):
            for seed in range(seeds if seeds is not None else (1 if quick else 3)):
                spec = dict(family=family, size=q, seed=seed,
                            depth=2 if family.startswith('layered') else depth_factor*q)
                yield 'full_reduce', spec, build(spec)


@contextmanager
def database_watchdog(seconds, interrupt):
    """Bound only database I/O and rewriting, never the reference or proofs."""
    expired = threading.Event()
    def expire():
        expired.set()
        interrupt()
    timer = threading.Timer(seconds, expire)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()
        if expired.is_set():
            raise TimeoutError(f'Database rewrite/I/O exceeded {seconds:g}s; interruption requested')


def trial(db, g, rule, gid, variant, order, *, unitary_input=False, certificate_timeout=0,
          certificate_cache=None, database_guard=None, evidence_directory=None,
          fixed_point_timeout=0, fixed_point_cache=None):
    """Same immutable input, fresh state, alternating engine order."""
    row = {}
    db.driver.last_query = None
    outputs = {}
    for engine in order:
        if engine == 'db':
            with database_guard() if database_guard else nullcontext():
                start = time.perf_counter()
                load(db, g, gid)
                loaded = time.perf_counter()
                db.driver.submissions = 0
                reduce_database(db, rule, gid, variant)
                reduced = time.perf_counter()
                row['queries'] = db.driver.submissions
                outputs[engine] = read(db, gid)
                row.update(db_import_s=loaded-start, db_rewrite_s=reduced-loaded,
                           db_export_s=time.perf_counter()-reduced)
                row['db_total_s'] = sum(row[k] for k in ('db_import_s', 'db_rewrite_s', 'db_export_s'))
        else:
            start = time.perf_counter()
            outputs[engine] = g.copy()
            copied = time.perf_counter()
            reference(outputs[engine], rule, time.monotonic()+120)
            row.update(pyzx_rewrite_s=time.perf_counter()-copied,
                       pyzx_total_s=time.perf_counter()-start)
    if evidence_directory is not None:
        row['graph_evidence'] = {name: save_graph(evidence_directory, graph)
                                 for name, graph in dict(original=g, **outputs).items()}
    # Verification is outside every timed region, against the original input.
    row['db_check'] = verify(g, outputs['db'])
    row['pyzx_check'] = verify(g, outputs['pyzx'])
    if unitary_input and certificate_timeout > 0:
        cache = certificate_cache if certificate_cache is not None else {}
        for engine in ('db', 'pyzx'):
            check = row[engine + '_check']
            if check['status'] != 'inconclusive':
                continue
            key = (digest(payload(g)), digest(payload(outputs[engine])))
            if key not in cache:
                cache[key] = certify_unitary_rewrite(g, outputs[engine], certificate_timeout)
            row[engine + '_check'] = dict(cache[key], tensor_check=check)
    if unitary_input and rule == 'full_reduce' and fixed_point_timeout > 0:
        cache = fixed_point_cache if fixed_point_cache is not None else {}
        for engine in ('db', 'pyzx'):
            key = digest(payload(outputs[engine]))
            if key not in cache:
                cache[key] = check_reduction_fixed_point(outputs[engine], fixed_point_timeout,
                                                        evidence_directory)
            row[engine + '_fixed_point'] = cache[key]
    row['db_output'] = metrics(outputs['db'])
    row['pyzx_output'] = metrics(outputs['pyzx'])
    row['db_changed'] = payload(g) != payload(outputs['db'])
    row['pyzx_changed'] = payload(g) != payload(outputs['pyzx'])
    row['status'] = ('failed' if any(row[k]['status'] == 'failed' for k in ('db_check', 'pyzx_check'))
                     else 'verified' if all(row[k]['status'] == 'verified' for k in ('db_check', 'pyzx_check'))
                     else 'inconclusive')
    return row


def summarize(rows):
    cells = {}
    for r in rows:
        if r['stage'] != 'benchmark':
            continue
        cells.setdefault((r['case'], r['variant']), []).append(r)
    summary = []
    for (case, variant), group in cells.items():
        good = [r for r in group if not r.get('warmup') and r.get('status') in ('verified', 'inconclusive')]
        complete = len(good) == group[0]['repetitions'] and all(r['status'] in ('verified', 'inconclusive') for r in group)
        complete = complete and len({r.get('attempt', 0) for r in good}) <= 1
        active = complete and all(r['db_changed'] and r['pyzx_changed'] for r in good)
        item = dict(case=case, rule=group[0]['rule'], variant=variant,
                    spec=group[0]['spec'], complete=complete, active=active,
                    statuses=dict(Counter(r['status'] for r in group)))
        for scope in ('rewrite', 'total'):
            # Never turn errors, failed semantics, warmups, or missing trials into wins.
            ratios = [r[f'pyzx_{scope}_s']/r[f'db_{scope}_s'] for r in good] if active else []
            item[scope+'_speedup'] = statistics.median(ratios) if ratios else None
            item[scope+'_range'] = [min(ratios), max(ratios)] if ratios else None
        item['verified'] = complete and all(r['status'] == 'verified' for r in good)
        for key in ('db_rewrite_s', 'pyzx_rewrite_s', 'db_total_s', 'pyzx_total_s'):
            item[key] = statistics.median(r[key] for r in good) if complete else None
        summary.append(item)
    return summary


def save_report(output, rows, environment, finished=False):
    summary = summarize(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['case', 'rule', 'variant', 'spec', 'complete', 'active', 'verified',
                  'statuses', 'rewrite_speedup', 'rewrite_range', 'total_speedup', 'total_range',
                  'db_rewrite_s', 'pyzx_rewrite_s', 'db_total_s', 'pyzx_total_s']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    counts = Counter(r['status'] for r in rows if r['stage'] == 'correctness')
    text = ['# ZXdb evaluation', '', f"Run state: {'finished' if finished else 'partial'}. Profile: {environment['profile']}.",
            '', '## Correctness', '', str(dict(counts)), '',
            'Each engine is checked against the original ordered tensor, up to a nonzero global scalar. '
            'Zero/nonzero disagreement fails. Tensor resource limits remain inconclusive unless the optional '
            'algebraic check reduces a known unitary circuit composition to ordered identity with a nonzero scalar. '
            'Proofs run outside the timing/watchdog region and are cached by input/output hashes.', '',
            '## Performance', '',
            f"{sum(r['complete'] for r in summary)}/{len(summary)} performance cells completed without failures. "
            f"{sum(bool(r['rewrite_speedup'] and r['rewrite_speedup'] > 1) for r in summary)} rewrite wins; "
            f"{sum(bool(r['total_speedup'] and r['total_speedup'] > 1) for r in summary)} total-time wins. "
            'Inspect tensor evidence per cell before drawing conclusions.', '',
            'Speedup = PyZX time / database time (>1 favors ZXdb). Median of paired ratios; '
            'ranges are observed variation, not confidence intervals. Warmups excluded. '
            'Both engines must change the input. Failed, incomplete, or mixed-attempt cells have no speedup.', '',
            'Rewrite timing excludes loading and reading. Total timing includes the evaluation adapter\'s '
            'in-memory graph import/export; PyZX total includes its graph copy. Neither includes disk I/O, '
            'process startup, connection setup, or verification. The legacy JSON importer/exporter is not benchmarked.', '',
            '| Workload | Variant | Rewrite speedup | Total speedup | Semantics verified |',
            '|---|---|---:|---:|---|']
    for r in summary:
        fmt = lambda v: '—' if v is None else f'{v:.3f}x'
        text.append(f"| {r['spec']['family']} n={r['spec']['size']} seed={r['spec']['seed']} | {r['variant']} | {fmt(r['rewrite_speedup'])} | {fmt(r['total_speedup'])} | {r['verified']} |")
    input_seeds = {}
    for row in rows:
        if row['stage'] == 'benchmark':
            key = (row['spec']['family'], row['spec']['size'], row['input_hash'])
            input_seeds.setdefault(key, set()).add(row['spec']['seed'])
    duplicates = [(family, size, sorted(seeds)) for (family, size, _), seeds in input_seeds.items()
                  if len(seeds) > 1]
    if duplicates:
        text += ['', '## Duplicate seed inputs', '',
                 'These seed settings generate identical input graphs; they are repeated measurements, not independent workloads.', '']
        text += [f'- {family} n={size}: seeds {seeds}' for family, size, seeds in duplicates]
    audits = {}
    for engine in ('db', 'pyzx'):
        checks = {r['graph_evidence'][engine]: r[engine + '_fixed_point'] for r in rows
                  if engine + '_fixed_point' in r and 'graph_evidence' in r}
        if checks:
            audits[engine] = dict(Counter(('changed' if c['structure_changed'] else 'unchanged')
                                          if c['status'] == 'completed' else 'inconclusive'
                                          for c in checks.values()))
    if audits:
        text += ['', '## Follow-up reference reduction', '',
                 'Counts below deduplicate exact saved outputs. This untimed structural check is separate from semantic equivalence.', '']
        text += [f'- {engine}: {counts}' for engine, counts in audits.items()]
    text += ['', '## Failures and unresolved evidence', '']
    for r in rows:
        if r['status'] in ('failed', 'error'):
            if r['stage'] == 'regressions':
                reason = f"Regression subprocess exit code {r.get('returncode')}; see regressions.log."
            elif r.get('error'):
                lines = r['error'].strip().splitlines()
                reason = (lines[-1] if lines[0].startswith('Traceback') else lines[0])[:300]
            else:
                reason = str({k: r.get(k) for k in ('db_check', 'pyzx_check')})
            text.append(f"- {r['stage']} / {r.get('rule', '')} / {r['case']} / {r.get('variant', '')}: {reason}")
    if environment.get('verification_review'):
        text += ['', '## Verification review', '', environment['verification_review']['description']]
    text += ['', '## Scope and interpretation', '',
             'Control is the ordinary Cypher implementation; optimized uses independent rewrite batches '
             'with exclusive access. Hadamard cancellation uses the same implementation '
             'in all variants. Native adds in-process C++ interior/boundary pivot, local complementation, gadget pivot and grouped supplementarity '
             'to the optimized schedule; remaining rules use Cypher. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense '
             'rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.', '',
             'Large cases may be unverified even when their small motifs pass. No capacity or correctness claim '
             'follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes '
             'are saved beside this report. Historical summaries are separate and are not evidence for this revision.', '',
             'Known limitations: database scalar tracking and legacy JSON import/export '
             'remain audit concerns. See '
             f'[audit history]({Path(os.path.relpath(ROOT / "evaluation" / "audit-history.md", output)).as_posix()}).']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='All correctness cases; smaller performance grid, one repeat')
    parser.add_argument('--correctness-only', action='store_true', help='Run regressions and all correctness fixtures without performance trials')
    parser.add_argument('--regression-timeout', type=float, default=300, help='Regression subprocess budget in seconds')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'latest')
    parser.add_argument('--repetitions', type=int, default=None)
    parser.add_argument('--variants', nargs='+', choices=['control', 'optimized', 'native'],
                        default=['control', 'optimized'])
    parser.add_argument('--full-reduce-only', action='store_true',
                        help='Benchmark circuits only; retain every correctness input')
    parser.add_argument('--qubits', nargs='+', type=int, help='Circuit sizes for the performance grid')
    parser.add_argument('--seeds', type=int, help='Number of deterministic circuit seeds')
    parser.add_argument('--depth-factor', type=int, default=10,
                        help='Random circuit depth per qubit (layered circuits use two layers)')
    parser.add_argument('--certificate-timeout', type=float, default=0,
                        help='Optional seconds per cached algebraic proof for circuit benchmark outputs; 0 disables')
    parser.add_argument('--fixed-point-timeout', type=float, default=0,
                        help='Optional seconds per cached follow-up PyZX reduction on circuit outputs; 0 disables')
    parser.add_argument('--resume', action='store_true', help='Continue an existing run; retain failures and skip finished cells')
    parser.add_argument('--restart-container', help='Dedicated Docker container to restart after database loss; failures remain recorded')
    parser.add_argument('--trial-timeout', type=float, default=120,
                        help='Rewrite/I/O database watchdog seconds when --restart-container is set')
    args = parser.parse_args()
    repetitions = args.repetitions if args.repetitions is not None else (1 if args.quick else 3)
    if repetitions < 1:
        parser.error('--repetitions must be positive')
    if args.fixed_point_timeout < 0:
        parser.error('--fixed-point-timeout must be nonnegative')
    if args.certificate_timeout < 0:
        parser.error('--certificate-timeout must be nonnegative')
    if args.trial_timeout <= 0:
        parser.error('--trial-timeout must be positive')
    if any(n < 1 for n in (args.qubits or []) + [args.seeds or 1, args.depth_factor]) or args.seeds == 0:
        parser.error('Circuit sizes, seeds and depth factor must be positive')
    if len(set(args.variants)) != len(args.variants):
        parser.error('--variants must not contain duplicates')
    os.chdir(ROOT)
    output = args.output.resolve()
    # A fresh folder prevents accidental mixing or destruction of prior evidence.
    output.mkdir(parents=True, exist_ok=True)
    if (output/'environment.json').exists() and not args.resume:
        parser.error('Output already contains a run; choose a fresh --output directory')
    if args.resume and not (output/'environment.json').exists():
        parser.error('--resume needs an existing output directory')
    logging.basicConfig(filename=output/'database.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    env = dict(started=datetime.now(timezone.utc).isoformat(), profile='quick' if args.quick else 'full',
               python=sys.version, platform=platform.platform(), cpu=platform.processor(),
               versions={p: importlib.metadata.version(p) for p in ('pyzx', 'neo4j', 'numpy', 'quimb')},
               repetitions=repetitions, warmups=1, variants=args.variants,
               benchmark_grid=dict(full_reduce_only=args.full_reduce_only, qubits=args.qubits,
                                   seeds=args.seeds, depth_factor=args.depth_factor),
               restart_container=args.restart_container,
               trial_timeout_s=args.trial_timeout, certificate_timeout_s=args.certificate_timeout,
               fixed_point_timeout_s=args.fixed_point_timeout,
               correctness_only=args.correctness_only, regression_timeout_s=args.regression_timeout,
               source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/'evaluate.py', *sorted((ROOT/'evaluation').glob('*.py')),
                                        *sorted((ROOT/'zxdb').glob('*.py')), *sorted((ROOT/'tests').glob('*.py')),
                                        ROOT/'zxdb/main_queries.json',
                                        ROOT/'zxdb_qm/zxqmcpp.cpp']})
    rows = []
    attempt = 0
    if args.resume:
        previous = json.loads((output/'environment.json').read_text(encoding='utf-8'))
        if previous.get('correctness_only', False) != args.correctness_only:
            parser.error('Resume configuration differs: correctness_only')
        for key in ('profile', 'repetitions', 'warmups', 'variants', 'versions', 'trial_timeout_s'):
            if previous[key] != env[key]:
                parser.error(f'Resume configuration differs: {key}')
        if previous.get('benchmark_grid', dict(full_reduce_only=False, qubits=None, seeds=None,
                                              depth_factor=10)) != env['benchmark_grid']:
            parser.error('Resume configuration differs: benchmark_grid')
        if previous.get('certificate_timeout_s', 0) != env['certificate_timeout_s']:
            parser.error('Resume configuration differs: certificate_timeout_s')
        if previous.get('fixed_point_timeout_s', 0) != env['fixed_point_timeout_s']:
            parser.error('Resume configuration differs: fixed_point_timeout_s')
        for name, digest_value in previous['source_hashes'].items():
            if name != 'evaluate.py' and env['source_hashes'].get(name) != digest_value:
                parser.error(f'Rewrite, oracle or workload source changed: {name}')
        rows = [json.loads(line) for line in (output/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
        history = previous.get('attempts', [dict(started=previous['started'], source_hashes=previous['source_hashes'])])
        attempt = len(history)
        env['attempts'] = history + [dict(started=env['started'], source_hashes=env['source_hashes'])]
    (output/f'runner-attempt-{attempt}.py').write_bytes(Path(__file__).read_bytes())
    with zipfile.ZipFile(output/f'sources-attempt-{attempt}.zip', 'x', compression=zipfile.ZIP_STORED) as archive:
        for name, expected in env['source_hashes'].items():
            data = (ROOT/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'Source changed during startup: {name}')
            archive.writestr(Path(name).as_posix(), data)
    def record(row):
        row['attempt'] = attempt
        rows.append(row)
        with (output/'observations.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        print(f"{row['stage']}: {row['case']} {row.get('variant', '')} {row['status']}", flush=True)
    (output/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    db = None
    finished = False
    gid = 'evaluation_' + uuid.uuid4().hex
    def connect():
        database = ZXdb(uri=uri, user=auth[0], password=auth[1])
        database._driver = CountingDriver(database.driver, timeout=30)
        return database

    def recover():
        """Optional experiment convenience, not a hidden retry of failed trials."""
        if not args.restart_container:
            raise RuntimeError('Database lost; use --restart-container only for a dedicated local Docker benchmark')
        if db is not None:
            db.close()
        restart = subprocess.run(['docker', 'restart', args.restart_container],
                                 stdout=subprocess.DEVNULL, timeout=30)
        for _ in range(30):
            try:
                database = connect()
                database.empty_graphdb(gid)
                record(dict(stage='recovery', case='database_restart', status='recovered',
                            restart_returncode=restart.returncode))
                return database
            except (ServiceUnavailable, SessionExpired):
                time.sleep(1)
        raise RuntimeError('Database did not recover within 30 connection attempts')

    certificate_cache = {}
    fixed_point_cache = {}

    def measured_trial(g, rule, variant, order, unitary_input=False):
        def interrupt_database():
            # Some analytical queries do not honor the driver's query timeout.
            subprocess.run(['docker', 'restart', args.restart_container], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        guard = (lambda: database_watchdog(args.trial_timeout, interrupt_database)) if args.restart_container else None
        return trial(db, g, rule, gid, variant, order, unitary_input=unitary_input,
                     certificate_timeout=args.certificate_timeout, certificate_cache=certificate_cache,
                     database_guard=guard, evidence_directory=output/'graphs',
                     fixed_point_timeout=args.fixed_point_timeout, fixed_point_cache=fixed_point_cache)
    try:
        uri = os.environ.get('ZXDB_URI', 'bolt://localhost:7687')
        auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
        with GraphDatabase.driver(uri, auth=auth) as driver:
            env['server'] = driver.get_server_info().agent
            with driver.session() as session:
                env['database_version'] = session.run('SHOW VERSION').data()
                if args.resume and previous.get('database_version') != env['database_version']:
                    raise RuntimeError('Database version changed since the previous attempt')
                if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                    raise RuntimeError('Use an empty dedicated Memgraph instance; no existing data is deleted')
                # This evaluator requires a dedicated empty instance. Remove
                # obsolete mutable-type indexes before regressions create graphs.
                for statement in ('DROP INDEX ON :Node(t)',
                                  'DROP EDGE INDEX ON :Wire(t)',
                                  'DROP INDEX ON :_ZXdbPivotWork',
                                  'DROP INDEX ON :_ZXdbLcompWork'):
                    session.run(statement).consume()
                env['indexes_before_run'] = session.run('SHOW INDEX INFO').data()
                if 'native' in env['variants']:
                    procedures = {r['name'] for r in session.run('CALL mg.procedures() YIELD name RETURN name')}
                    required = {'zxqmcpp.pivot', 'zxqmcpp.lcomp', 'zxqmcpp.gadget',
                                'zxqmcpp.supplementarity', 'zxqmcpp.pauli_boundary', 'zxqmcpp.boundary'}
                    if not required <= procedures:
                        raise RuntimeError('Compile and load the native module first; see zxdb_qm/README.md')
        # Run implementation regressions as part of the same pipeline, saving their output.
        if not any(r['stage'] == 'regressions' for r in rows):
            with (output/'regressions.log').open('w', encoding='utf-8') as log:
                modules = ['test_evaluation', 'test_evidence', 'test_graph_isolation', 'test_hopf_normalization',
                           'test_pivot_materialization', 'test_gadget_pivot_batch',
                           'test_boundary_pivot_batch', 'test_reduction_optimizations', 'test_rule_audit']
                if 'native' in env['variants']:
                    modules.append('test_native_reduction')
                if os.environ.get('ZXDB_PYTHON_MODULE') == '1':
                    modules.append('test_python_gadget')
                try:
                    result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                                             *['tests.' + name for name in modules]],
                                            stdout=log, stderr=subprocess.STDOUT, timeout=args.regression_timeout,
                                            env=dict(os.environ, ZXDB_LIVE='1'))
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    log.write(f'\nRegression subprocess exceeded {args.regression_timeout:g} seconds and was stopped.\n')
                    code = -1
            record(dict(stage='regressions', case='unittest', status='verified' if code == 0 else 'failed',
                        returncode=code))
        try:
            db = connect()
        except (ServiceUnavailable, SessionExpired):
            db = recover()
        for rule, name, g in correctness_cases():
            for variant in env['variants']:
                if any(r['stage'] == 'correctness' and (r['rule'], r['case'], r['variant']) == (rule, name, variant) for r in rows):
                    continue
                row = dict(stage='correctness', case=name, rule=rule, variant=variant,
                           input_hash=digest(payload(g)), input=metrics(g))
                try:
                    with redirect_stdout(io.StringIO()):
                        row.update(measured_trial(g, rule, variant, ('pyzx', 'db')))
                except Exception as exc:
                    row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                               last_query=db.driver.last_query)
                    if isinstance(exc, (ServiceUnavailable, SessionExpired)) or (isinstance(exc, TimeoutError) and args.restart_container):
                        record(row)
                        db = recover()
                        continue
                record(row)
            save_report(output, rows, env)
        for rule, spec, g in (() if args.correctness_only else benchmark_cases(args.quick, **env['benchmark_grid'])):
            for variant in env['variants']:
                prior = [r for r in rows if r['stage'] == 'benchmark' and r['case'] == digest(spec)[:16] and r['variant'] == variant]
                if any(r['status'] == 'error' for r in prior):
                    continue
                for i in range(repetitions+1):
                    if any(r['repetition'] == i for r in prior):
                        continue
                    row = dict(stage='benchmark', case=digest(spec)[:16], spec=spec, rule=rule,
                               variant=variant, repetition=i, warmup=i == 0, repetitions=repetitions,
                               input_hash=digest(payload(g)), input=metrics(g))
                    try:
                        with redirect_stdout(io.StringIO()):
                            row.update(measured_trial(g, rule, variant,
                                             ('db', 'pyzx') if i % 2 else ('pyzx', 'db'),
                                             unitary_input=rule == 'full_reduce'))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                               last_query=db.driver.last_query)
                        if isinstance(exc, (ServiceUnavailable, SessionExpired)) or (isinstance(exc, TimeoutError) and args.restart_container):
                            record(row)
                            db = recover()
                            break
                    record(row)
                    if row['status'] == 'error':
                        break
                save_report(output, rows, env)
        finished = True
    except Exception:
        record(dict(stage='pipeline', case='execution', status='error', error=traceback.format_exc()))
    finally:
        if db is not None:
            try:
                db.empty_graphdb(gid)
            except Exception as exc:
                record(dict(stage='pipeline', case='cleanup', status='error', error=str(exc)))
            db.close()
        (output/'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
        save_report(output, rows, env, finished=finished)
    return 1 if any(r['status'] in ('failed', 'error') for r in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
