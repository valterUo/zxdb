"""Bounded diagnostic of database variants, not a publication benchmark.

Uses 16/64/128 disjoint motifs and one modest realistic circuit, one warmup and
three repetitions per mode. Use research campaigns for population estimates.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import uuid

from neo4j import GraphDatabase
from zxdb.zxdb import ZXdb
from evaluation.research import reduce_database
from evaluation.research_core import CountingDriver, load, read, verify
from evaluation.research_workloads import build, metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    uri = os.environ['ZXDB_URI']
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as s:
            if s.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                raise RuntimeError('Requires an empty dedicated database')
    db = ZXdb(uri=uri, user=auth[0], password=auth[1])
    gid = 'optimization_probe_' + uuid.uuid4().hex
    raw_driver = db.driver
    rows = []
    try:
        cases = [(f'copies_{r}', r, n, None) for r in ('pivot', 'lcomp') for n in (16, 64, 128)]
        cases += [('random_clifford_t', 'full_reduce', 4, 40)]
        for family, rule, size, depth in cases:
            graph = build(dict(family=family, size=size, depth=depth, seed=0))
            by_variant = {v: [] for v in ('control', 'optimized', 'worklist')}
            for trial in range(-1, 3):
                order = list(by_variant)
                if trial % 2:
                    order.reverse()
                for variant in order:
                    print(f'Attempt {family} {size} {variant} trial={trial}: load', flush=True)
                    load(db, graph, gid)
                    counter = CountingDriver(raw_driver, timeout=30, deadline=time.monotonic()+120)
                    db._driver = counter
                    try:
                        print('reduce', flush=True)
                        t = time.perf_counter(); reduce_database(db, rule, gid, variant)
                        elapsed = time.perf_counter()-t
                    finally:
                        db._driver = raw_driver
                    result = read(db, gid)
                    check = verify(graph, result)
                    if check['status'] == 'failed':
                        raise AssertionError(check)
                    if trial >= 0:
                        by_variant[variant].append(dict(seconds=elapsed, submissions=counter.submissions,
                                                        output=metrics(result), verification=check))
                    print('complete', flush=True)
            for variant, trials in by_variant.items():
                row = dict(family=family, size=size, depth=depth, variant=variant, trials=trials,
                           median_s=statistics.median(t['seconds'] for t in trials),
                           submissions=statistics.median(t['submissions'] for t in trials))
                rows.append(row)
                print(f'{family} {size} {variant}: {row["median_s"]:.6f}s, {row["submissions"]} calls', flush=True)
    finally:
        db._driver = raw_driver
        Path(args.output).write_text(json.dumps(rows, indent=2))
        try:
            db.empty_graphdb(gid)
        finally:
            db.close()


if __name__ == '__main__':
    main()
