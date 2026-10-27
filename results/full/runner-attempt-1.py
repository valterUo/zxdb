"""Run the complete audit and paired PyZX benchmark: python evaluate.py."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
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

import pyzx as zx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from evaluation.cases import RULES as CASES
from evaluation.circuits import build_suite
from evaluation.core import CountingDriver, load, read, reference, verify
from evaluation.workloads import RULES, build, digest, metrics, payload
from zxdb.zxdb import ZXdb

ROOT = Path(__file__).resolve().parent


def reduce_database(db, rule, gid, variant):
    method = RULES[rule][0]
    if variant == 'control' or rule == 'hadamard_cancel':
        return getattr(db, method)(gid)
    if rule == 'full_reduce':
        return db.full_reduce_optimized(gid, exclusive_access=True)
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


def benchmark_cases(quick):
    """Predeclared scales: independent matches, fan growth, realistic circuits."""
    for rule in RULES:
        if rule == 'full_reduce':
            continue
        for n in ([2, 8] if quick else [2, 32, 512]):
            spec = dict(family='copies_' + rule, size=n, seed=0)
            yield rule, spec, build(spec)
        for n in ([2] if quick else [2, 16, 128]):
            spec = dict(family=rule, size=n, seed=0)
            yield rule, spec, build(spec)
    for family in ('random_clifford', 'random_clifford_t', 'random_t_heavy',
                   'layered_local', 'layered_global'):
        for q in ([3] if quick else [3, 8, 16]):
            for seed in range(1 if quick else 3):
                spec = dict(family=family, size=q, seed=seed,
                            depth=2 if family.startswith('layered') else 10*q)
                yield 'full_reduce', spec, build(spec)


def trial(db, g, rule, gid, variant, order):
    """Same immutable input, fresh state, alternating engine order."""
    row = {}
    outputs = {}
    for engine in order:
        start = time.perf_counter()
        if engine == 'db':
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
            outputs[engine] = g.copy()
            copied = time.perf_counter()
            reference(outputs[engine], rule, time.monotonic()+120)
            row.update(pyzx_rewrite_s=time.perf_counter()-copied,
                       pyzx_total_s=time.perf_counter()-start)
    # Verification is outside every timed region, against the original input.
    row['db_check'] = verify(g, outputs['db'])
    row['pyzx_check'] = verify(g, outputs['pyzx'])
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
            'Zero/nonzero disagreement fails. Tensor resource limits are inconclusive, never passes.', '',
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
            '| Workload | Variant | Rewrite speedup | Total speedup | Tensor verified |',
            '|---|---|---:|---:|---|']
    for r in summary:
        fmt = lambda v: '—' if v is None else f'{v:.3f}x'
        text.append(f"| {r['spec']['family']} n={r['spec']['size']} seed={r['spec']['seed']} | {r['variant']} | {fmt(r['rewrite_speedup'])} | {fmt(r['total_speedup'])} | {r['verified']} |")
    text += ['', '## Failures and unresolved evidence', '']
    for r in rows:
        if r['status'] in ('failed', 'error'):
            reason = r.get('error') or str({k: r.get(k) for k in ('db_check', 'pyzx_check')})
            text.append(f"- {r['stage']} / {r.get('rule', '')} / {r['case']} / {r.get('variant', '')}: {reason}")
    text += ['', '## Scope and interpretation', '',
             'Control is the ordinary Cypher implementation; optimized uses independent rewrite batches '
             'with exclusive access and no incremental worklist. Hadamard cancellation uses the same implementation '
             'in both variants. Synthetic disjoint copies isolate batching opportunities; fan patterns test dense '
             'rewrites; random and layered circuits test full_reduce. These are distinct workloads, not a universal speedup.', '',
             'Large cases may be unverified even when their small motifs pass. No capacity or correctness claim '
             'follows from a timeout or an inconclusive check. Raw observations, input hashes, versions and source hashes '
             'are saved beside this report. Historical summaries are separate and are not evidence for this revision.', '',
             'Known limitations: database scalar tracking, legacy JSON import/export, and standalone '
             'supplementarity preconditions remain audit concerns. See ../../evaluation/audit-history.md.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='All correctness cases; smaller performance grid, one repeat')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'latest')
    parser.add_argument('--repetitions', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Continue an existing run; retain failures and skip finished cells')
    parser.add_argument('--restart-container', help='Dedicated Docker container to restart after database loss; failures remain recorded')
    parser.add_argument('--trial-timeout', type=float, default=120,
                        help='Whole-trial database watchdog seconds when --restart-container is set')
    args = parser.parse_args()
    repetitions = args.repetitions if args.repetitions is not None else (1 if args.quick else 3)
    if repetitions < 1:
        parser.error('--repetitions must be positive')
    if args.trial_timeout <= 0:
        parser.error('--trial-timeout must be positive')
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
               repetitions=repetitions, warmups=1, variants=['control', 'optimized'],
               restart_container=args.restart_container,
               trial_timeout_s=args.trial_timeout,
               source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/'evaluate.py', *sorted((ROOT/'evaluation').glob('*.py')),
                                        *sorted((ROOT/'zxdb').glob('*.py')), ROOT/'zxdb/main_queries.json']})
    rows = []
    attempt = 0
    if args.resume:
        previous = json.loads((output/'environment.json').read_text(encoding='utf-8'))
        for key in ('profile', 'repetitions', 'warmups', 'variants', 'versions', 'trial_timeout_s'):
            if previous[key] != env[key]:
                parser.error(f'Resume configuration differs: {key}')
        for name, digest_value in previous['source_hashes'].items():
            if name != 'evaluate.py' and env['source_hashes'].get(name) != digest_value:
                parser.error(f'Rewrite, oracle or workload source changed: {name}')
        rows = [json.loads(line) for line in (output/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
        history = previous.get('attempts', [dict(started=previous['started'], source_hashes=previous['source_hashes'])])
        attempt = len(history)
        env['attempts'] = history + [dict(started=env['started'], source_hashes=env['source_hashes'])]
    (output/f'runner-attempt-{attempt}.py').write_bytes(Path(__file__).read_bytes())
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

    def measured_trial(g, rule, variant, order):
        expired = threading.Event()
        def interrupt_database():
            expired.set()
            # The driver timeout is not sufficient for some analytical queries.
            subprocess.run(['docker', 'restart', args.restart_container], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        timer = threading.Timer(args.trial_timeout, interrupt_database) if args.restart_container else None
        if timer:
            timer.start()
        try:
            return trial(db, g, rule, gid, variant, order)
        finally:
            if timer:
                timer.cancel()
                timer.join()
            if expired.is_set():
                raise TimeoutError(f'Trial exceeded {args.trial_timeout:g}s; database interrupted')
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
        # Run implementation regressions as part of the same pipeline, saving their output.
        if not any(r['stage'] == 'regressions' for r in rows):
            with (output/'regressions.log').open('w', encoding='utf-8') as log:
                modules = ['test_evaluation', 'test_graph_isolation', 'test_hopf_normalization',
                           'test_pivot_materialization', 'test_gadget_pivot_batch',
                           'test_boundary_pivot_batch', 'test_reduction_optimizations']
                try:
                    result = subprocess.run([sys.executable, '-m', 'unittest', '-v',
                                             *['tests.' + name for name in modules]],
                                            stdout=log, stderr=subprocess.STDOUT, timeout=300,
                                            env=dict(os.environ, ZXDB_LIVE='1'))
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    log.write('\nRegression subprocess exceeded 300 seconds and was stopped.\n')
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
                    row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    if isinstance(exc, (ServiceUnavailable, SessionExpired)) or (isinstance(exc, TimeoutError) and args.restart_container):
                        record(row)
                        db = recover()
                        continue
                record(row)
            save_report(output, rows, env)
        for rule, spec, g in benchmark_cases(args.quick):
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
                                             ('db', 'pyzx') if i % 2 else ('pyzx', 'db')))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
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
