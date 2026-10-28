"""Resumable current-revision three-engine campaign, with untimed comparisons."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid
import zipfile

import pyzx
from neo4j import GraphDatabase

from evaluate import benchmark_cases, database_watchdog, reduce_database
from evaluation.core import CountingDriver, load, read, reference, verify, compare_outputs
from evaluation.evidence import save_graph
from evaluation.workloads import build, digest, metrics, payload
from zxdb.zxdb import ZXdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--qubits', nargs='+', type=int, default=[3, 8, 16, 32, 64, 128])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    uri = os.environ.get('ZXDB_URI', 'bolt://127.0.0.1:7691')
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as session:
            if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                raise RuntimeError('Requires an empty dedicated database')
            server = session.run('SHOW VERSION').data()
    jobs = [(rule, config) for rule, config, _ in benchmark_cases(
        False, qubits=args.qubits, seeds=args.seeds, depth_factor=20)]
    jobs += [('identity', dict(family='identity_chain', size=n, seed=0)) for n in (2, 16, 128, 512)]
    sources = [root/'evaluate.py', *sorted((root/'evaluation').glob('*.py')),
               *sorted((root/'zxdb').glob('*.py')), root/'zxdb/main_queries.json', root/'zxdb_qm/zxqmcpp.cpp']
    metadata = dict(manifest=[dict(rule=r, spec=s) for r, s in jobs],
                    repetitions=args.repetitions, warmups=1, uri=uri, server=server,
                    python=sys.version, platform=platform.platform(), pyzx=pyzx.__version__,
                    sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    order='deterministic shuffle per configuration and repetition',
                    shared_pyzx=True, depth_factor=20, layered_depth=2)
    envfile = output/'environment.json'
    if envfile.exists() and json.loads(envfile.read_text()) != metadata:
        raise RuntimeError('Sources or configuration changed; choose a new output directory')
    envfile.write_text(json.dumps(metadata, indent=2)+'\n')
    if not (output/'sources.zip').exists():
        with zipfile.ZipFile(output/'sources.zip', 'x') as archive:
            for path in sources:
                archive.write(path, path.relative_to(root))
    logfile = output/'observations.jsonl'
    rows = [json.loads(line) for line in logfile.read_text().splitlines()] if logfile.exists() else []
    done = {(r['case'], r['repetition']) for r in rows if r['engine'] == 'comparison_complete'}
    if rows and any((r['case'], r['repetition']) not in done for r in rows):
        raise RuntimeError('Interrupted repetition: preserve it and use a fresh output directory')
    db = ZXdb(uri=uri, user=auth[0], password=auth[1])
    db._driver = CountingDriver(db.driver, timeout=300)
    gid = 'three_engine_' + uuid.uuid4().hex
    cache = {}
    def record(row):
        with logfile.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps({k: row[k] for k in ('case', 'engine', 'repetition', 'status', 'rewrite_s') if k in row}), flush=True)
    def interrupt():
        subprocess.run(['docker', 'restart', args.container], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for index, (rule, spec) in enumerate(jobs):
            g = build(spec)
            case = digest(spec)[:16]
            original_hash = save_graph(output/'graphs', g)
            print(f'WORKLOAD {index+1}/{len(jobs)} {spec}', flush=True)
            for repetition in range(args.repetitions+1):
                if (case, repetition) in done:
                    continue
                order = ['pyzx', 'cypher', 'hybrid']
                random.Random(index*17+repetition).shuffle(order)
                results, batch = {}, []
                for engine in order:
                    row = dict(case=case, rule=rule, spec=spec, engine=engine,
                               repetition=repetition, warmup=repetition == 0,
                               input_hash=original_hash, input=metrics(g))
                    try:
                        if engine == 'pyzx':
                            start = time.perf_counter()
                            result = g.copy()
                            copied = time.perf_counter()
                            with redirect_stdout(io.StringIO()):
                                reference(result, rule, time.monotonic()+300)
                            end = time.perf_counter()
                            row.update(rewrite_s=end-copied, total_s=end-start)
                        else:
                            with database_watchdog(300, interrupt):
                                start = time.perf_counter()
                                load(db, g, gid)
                                loaded = time.perf_counter()
                                db.driver.submissions = 0
                                with redirect_stdout(io.StringIO()):
                                    reduce_database(db, rule, gid, 'native' if engine == 'hybrid' else 'optimized')
                                reduced = time.perf_counter()
                                row['queries'] = db.driver.submissions
                                result = read(db, gid)
                                end = time.perf_counter()
                            row.update(import_s=loaded-start, rewrite_s=reduced-loaded,
                                       export_s=end-reduced, total_s=end-start)
                            db.empty_graphdb(gid)
                        results[engine] = result
                        result_hash = save_graph(output/'graphs', result)
                        key = ('input', original_hash, result_hash)
                        if key not in cache:
                            cache[key] = verify(g, result)
                        row.update(check=cache[key], status=cache[key]['status'],
                                   output=metrics(result), changed=payload(g) != payload(result),
                                   graphs=dict(original=original_hash, result=result_hash))
                    except Exception as exc:
                        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                        if engine != 'pyzx':
                            db.empty_graphdb(gid)
                    batch.append(row)
                for row in batch:
                    engine = row['engine']
                    if engine != 'pyzx' and engine in results and 'pyzx' in results:
                        ref_hash = next(r['graphs']['result'] for r in batch if r['engine'] == 'pyzx')
                        key = ('reference', row['graphs']['result'], ref_hash)
                        if key not in cache:
                            cache[key] = compare_outputs(results[engine], results['pyzx'],
                                                         max_iso_vertices=2048, iso_timeout_s=2)
                        row['against_pyzx'] = cache[key]
                        if cache[key]['tensor']['status'] == 'failed':
                            row['status'] = 'failed'
                    record(row)
                record(dict(case=case, engine='comparison_complete', repetition=repetition, status='recorded'))
        (output/'completion.json').write_text(json.dumps(dict(finished=True, configurations=len(jobs),
                                                             trials=len(jobs)*3*(args.repetitions+1)), indent=2))
    finally:
        db.empty_graphdb(gid)
        db.close()


if __name__ == '__main__':
    main()
