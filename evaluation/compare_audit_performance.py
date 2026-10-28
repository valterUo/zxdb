"""Sequential, resumable five-engine comparison with one shared timed PyZX run.

Used for large circuits and supplemental workloads after the standard evaluator.
No reference timing is cached across repetitions. Pre-fix sources are immutable
git exports, and all engines receive fresh copies of the same original graph.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

import pyzx as zx
from neo4j import GraphDatabase

from evaluate import benchmark_cases, database_watchdog, reduce_database
from evaluation.core import CountingDriver, load, read, reference, verify
from evaluation.evidence import save_graph
from evaluation.workloads import build, digest, metrics, payload
from zxdb.zxdb import ZXdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--container', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign = args.campaign.resolve()
    output = campaign / 'comparison'
    output.mkdir(exist_ok=True)
    before = campaign / 'pre-fix-source'
    for name in ('zxdb_qm/zxqmcpp.cpp', 'zxdb/pyzx_utils.py'):
        if (before/name).read_bytes().replace(b'\r\n', b'\n') != (root/name).read_bytes().replace(b'\r\n', b'\n'):
            raise RuntimeError(f'Comparison needs a separate pre-fix dependency: {name}')
    spec = importlib.util.spec_from_file_location('pre_audit_zxdb', before / 'zxdb/zxdb.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_queries = json.loads((before / 'zxdb/main_queries.json').read_text())['items']
    uri = os.environ.get('ZXDB_URI', 'bolt://127.0.0.1:7691')
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    # Never silently clear a different task's database.
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as session:
            if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                raise RuntimeError('Comparison requires an empty dedicated database')
    databases = {}
    for name in ('cypher', 'hybrid', 'before_cypher', 'before_hybrid'):
        db = (module.ZXdb if name.startswith('before_') else ZXdb)(uri=uri, user=auth[0], password=auth[1])
        if name.startswith('before_'):
            db.basic_rewrite_rule_queries = {item['title']: item for item in old_queries}
        db._driver = CountingDriver(db.driver, timeout=120)
        databases[name] = db
    jobs = []
    for rule, config, _ in benchmark_cases(False, full_reduce_only=True,
                                          qubits=[32, 64, 128], seeds=3, depth_factor=20):
        jobs.append((rule, config, ('pyzx', *databases)))
    for n in (2, 16, 128, 512):
        jobs.append(('identity', dict(family='identity_chain', size=n, seed=0), ('pyzx', *databases)))
    # The standard evaluator already measured current code and PyZX. Only the
    # pre-fix database paths are missing for that identical workload manifest.
    for rule, config, _ in benchmark_cases(False):
        jobs.append((rule, config, ('before_cypher', 'before_hybrid')))
    manifest = [dict(rule=r, spec=s, engines=list(e)) for r, s, e in jobs]
    fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root/'zxdb/zxdb.py', root/'zxdb/main_queries.json',
                              root/'zxdb_qm/zxqmcpp.cpp', root/'evaluate.py',
                              root/'evaluation/core.py', root/'evaluation/workloads.py',
                              root/'evaluation/patterns.py', root/'evaluation/oracle.py',
                              before/'zxdb/zxdb.py', before/'zxdb/main_queries.json',
                              Path(__file__))}
    metadata = dict(manifest=manifest, repetitions=3, warmups=1,
                    sources=fingerprints, uri=uri, shared_pyzx=True,
                    order='deterministic shuffle for each repetition',
                    note='PyZX timed once per spec/repetition, shared by all four DB paths')
    envfile = output/'environment.json'
    if envfile.exists() and json.loads(envfile.read_text()) != metadata:
        raise RuntimeError('Source or manifest changed; use a new campaign directory')
    envfile.write_text(json.dumps(metadata, indent=2)+'\n')
    logfile = output/'observations.jsonl'
    rows = [json.loads(line) for line in logfile.read_text().splitlines()] if logfile.exists() else []
    done = {(r['case'], r['engine'], r['repetition']) for r in rows}
    gid = 'audit_benchmark_' + uuid.uuid4().hex
    failures = set((r['case'], r['engine']) for r in rows if r['status'] == 'error')
    def interrupt():
        subprocess.run(['docker', 'restart', args.container], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for job_index, (rule, config, engines) in enumerate(jobs):
            g = build(config)
            case = digest(config)[:16]
            for repetition in range(4):
                order = list(engines)
                random.Random(job_index*17 + repetition).shuffle(order)
                for engine in order:
                    if (case, engine, repetition) in done or (case, engine) in failures:
                        continue
                    row = dict(case=case, rule=rule, spec=config, engine=engine,
                               repetition=repetition, warmup=repetition == 0,
                               input_hash=digest(payload(g)), input=metrics(g))
                    cleanup_error = None
                    try:
                        if engine == 'pyzx':
                            start = time.perf_counter()
                            result = g.copy()
                            copied = time.perf_counter()
                            # Same stock reference as evaluate.py. Its single
                            # full_reduce call is not interrupted at 120s.
                            with redirect_stdout(io.StringIO()):
                                reference(result, rule, time.monotonic()+120)
                            end = time.perf_counter()
                            row.update(rewrite_s=end-copied, total_s=end-start)
                        else:
                            db = databases[engine]
                            with database_watchdog(120, interrupt):
                                start = time.perf_counter()
                                load(db, g, gid)
                                loaded = time.perf_counter()
                                db.driver.submissions = 0
                                with redirect_stdout(io.StringIO()):
                                    reduce_database(db, rule, gid,
                                                    'native' if engine.endswith('hybrid') else 'optimized')
                                reduced = time.perf_counter()
                                row['queries'] = db.driver.submissions
                                result = read(db, gid)
                                end = time.perf_counter()
                                row.update(import_s=loaded-start, rewrite_s=reduced-loaded,
                                           export_s=end-reduced, total_s=end-start)
                            db.empty_graphdb(gid)
                        # All checks and evidence creation are outside timers.
                        row['check'] = verify(g, result)
                        row['status'] = row['check']['status']
                        row['output'] = metrics(result)
                        row['changed'] = payload(g) != payload(result)
                        row['graphs'] = dict(original=save_graph(output/'graphs', g),
                                             result=save_graph(output/'graphs', result))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                        failures.add((case, engine))
                        if engine != 'pyzx':
                            try:
                                databases[engine].empty_graphdb(gid)
                            except Exception as cleanup_exc:
                                cleanup_error = str(cleanup_exc)
                                row['cleanup_error'] = cleanup_error
                    with logfile.open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    rows.append(row)
                    print(f'{job_index+1}/{len(jobs)} {config} {engine} rep={repetition} '
                          f"{row['status']} rewrite={row.get('rewrite_s')}", flush=True)
                    if cleanup_error is not None:
                        raise RuntimeError('Database unavailable after recorded failure: '+cleanup_error)
        (output/'completion.json').write_text(json.dumps(dict(
            finished=True, statuses=dict(Counter(r['status'] for r in rows)),
            recorded=len(rows), planned=sum(len(e)*4 for _, _, e in jobs)), indent=2)+'\n')
    finally:
        for db in databases.values():
            db.close()


if __name__ == '__main__':
    main()
