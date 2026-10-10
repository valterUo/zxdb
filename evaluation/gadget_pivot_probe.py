"""Bounded gadget/boundary-pivot diagnostic, not a publication campaign.

Requires an empty dedicated ZXDB_URI. Three motif mixtures per size, one
excluded warmup and two measured repetitions. Timings exclude load/read/copy.
Each case runs in a subprocess with a hard timeout; failures stop the probe.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid

import pyzx
from evaluation.research_core import CountingDriver, compare_outputs, load, read, reference
from evaluation.research_workloads import build, digest, metrics, payload
from zxdb.zxdb import ZXdb


def measure(size, seed, gid=None, rule='pivot_gadget', control_max_size=128):
    # Different proportions yield genuinely different graph structures.
    triples = size * (seed + 1) // 4
    spec = dict(family='copies_'+rule, size=size, seed=seed,
                motif_counts={'2': size-triples, '3': triples})
    graph = build(spec)
    gid = gid or 'gadget_probe_' + uuid.uuid4().hex
    variants = ['pyzx', 'optimized'] + (['control'] if size <= control_max_size else [])
    rows = []
    with ZXdb(uri=os.environ['ZXDB_URI'], user=os.environ.get('ZXDB_USER',''),
              password=os.environ.get('ZXDB_PASSWORD','')) as db:
        with db.driver.session() as session:
            if session.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                raise RuntimeError('Requires an empty dedicated database')
        raw = db.driver
        try:
            for trial in range(-1, 2):
                order = variants.copy()
                random.Random(f'{size}:{seed}:{trial}').shuffle(order)
                outputs = {}
                for variant in order:
                    if variant == 'pyzx':
                        result = graph.copy()
                        start = time.perf_counter()
                        reference(result, rule, time.monotonic()+15)
                        elapsed = time.perf_counter()-start
                        submissions = 0
                    else:
                        load(db, graph, gid)
                        counter = CountingDriver(raw, timeout=15, deadline=time.monotonic()+20)
                        db._driver = counter
                        try:
                            start = time.perf_counter()
                            if variant == 'optimized':
                                db.apply_rule_optimized(rule+'_rule', gid, exclusive_access=True)
                            else:
                                getattr(db,rule+'_rule')(gid)
                            elapsed = time.perf_counter()-start
                        finally:
                            db._driver = raw
                        submissions = counter.submissions
                        result = read(db, gid)
                    outputs[variant] = result
                    rows.append(dict(trial=trial, warmup=trial < 0, variant=variant,
                                     seconds=elapsed, submissions=submissions))
                for variant in variants:
                    if variant == 'pyzx':
                        continue
                    check = compare_outputs(outputs[variant], outputs['pyzx'])
                    if check['tensor']['status'] == 'failed' or check['degree_sequence']['status'] != 'equal':
                        raise AssertionError(check)
                    next(r for r in rows if r['trial'] == trial and r['variant'] == variant)['validation'] = check
        finally:
            db._driver = raw
            db.empty_graphdb(gid)
    return dict(spec=spec, input_hash=digest(payload(graph)), input=metrics(graph), trials=rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output')
    parser.add_argument('--budget-seconds', type=float, default=180)
    parser.add_argument('--case-size', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--graph-id', help=argparse.SUPPRESS)
    parser.add_argument('--rule',choices=('pivot_gadget','pivot_boundary'),default='pivot_gadget')
    parser.add_argument('--control-max-size',type=int,default=128)
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,64,128,256,512,1024,2048])
    args = parser.parse_args()
    if args.case_size:
        print(json.dumps(measure(args.case_size,args.seed,args.graph_id,args.rule,args.control_max_size)))
        return
    if not args.output:
        parser.error('--output is required')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output file; historical results are preserved')
    output.parent.mkdir(parents=True,exist_ok=True)
    report = dict(python=platform.python_version(), pyzx=pyzx.__version__,
                  sizes=args.sizes, seeds=[0,1,2],
                  rule=args.rule, warmups=1, repetitions=2,
                  control_max_size=args.control_max_size, cases=[])
    report['source_hashes'] = {name: digest(Path(name).read_text()) for name in
                             ('zxdb/main_queries.json','zxdb/zxdb.py',__file__)}
    deadline = time.monotonic()+args.budget_seconds
    try:
        for size in report['sizes']:
            for seed in report['seeds']:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Probe budget exhausted')
                gid = 'gadget_probe_' + uuid.uuid4().hex
                try:
                    result = subprocess.run([sys.executable,'-m','evaluation.gadget_pivot_probe',
                                             '--case-size',str(size),'--seed',str(seed),
                                             '--rule',args.rule,'--control-max-size',str(args.control_max_size),
                                             '--graph-id',gid], capture_output=True, text=True,
                                            timeout=min(60,remaining))
                except subprocess.TimeoutExpired as exc:
                    report['status'] = 'budget_exhausted' if remaining < 60 else 'case_timeout'
                    report['unfinished_case'] = dict(size=size,seed=seed)
                    stderr=exc.stderr or ''
                    report['worker_stderr'] = (stderr.decode(errors='replace') if isinstance(stderr,bytes) else stderr)[-8000:]
                    # A killed child cannot execute its finally block. Clean up
                    # only the graph ID allocated for that child, never the DB.
                    try:
                        with ZXdb(uri=os.environ['ZXDB_URI'],user=os.environ.get('ZXDB_USER',''),
                                  password=os.environ.get('ZXDB_PASSWORD','')) as db:
                            db.empty_graphdb(gid)
                    except Exception as cleanup_error:
                        report['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
                        report['status'] = 'timeout_and_cleanup_failed'
                    return
                if result.returncode:
                    raise RuntimeError(result.stderr[-4000:])
                case = json.loads(result.stdout)
                report['cases'].append(case)
                medians = {v: statistics.median(r['seconds'] for r in case['trials']
                           if r['variant'] == v and not r['warmup']) for v in ('pyzx','optimized')}
                print(size,seed,medians,'speedup',medians['pyzx']/medians['optimized'],flush=True)
                output.write_text(json.dumps(report,indent=2))
        report['status'] = 'complete'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
