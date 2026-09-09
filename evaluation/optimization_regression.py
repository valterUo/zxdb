"""Run every existing adversarial rule/full-reduce case on a dedicated server.

ZXDB_URI must explicitly select the server. Scoped in-memory load/read avoids
the historical harness's global importer and filesystem graph exports.
"""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from neo4j import GraphDatabase
from zxdb.zxdb import ZXdb
from evaluation import harness
from evaluation.cases import RULES
from evaluation.eval_full_reduce import build_suite, pyzx_full_reduce
from evaluation.research_core import load, read


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--variant', default='full_reduce')
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclusive-access', action='store_true')
    parser.add_argument('--incremental', action='store_true')
    args = parser.parse_args()
    uri = os.environ['ZXDB_URI']
    auth = (os.environ.get('ZXDB_USER', ''), os.environ.get('ZXDB_PASSWORD', ''))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as s:
            if s.run('MATCH(n) RETURN count(n) AS n').single()['n']:
                raise RuntimeError('Requires an empty dedicated database')
    db = ZXdb(uri=uri, user=auth[0], password=auth[1])
    results = []
    def record(result):
        results.append(result)
        print(result['rule'], result['case'], result['ok'], result.get('level'), flush=True)
        Path(args.output).write_text(json.dumps(dict(variant=args.variant, results=results), indent=2))
        if result.get('error'):
            print(result['error'], flush=True)
            raise RuntimeError('Regression stopped after a worker/database error; checkpoint saved')
    def importer(db, graph, graph_id):
        load(db, graph, graph_id)
    def exporter(self, graph_id, *args, **kwargs):
        return read(self, graph_id)
    started = time.monotonic()
    try:
        with patch.object(harness, 'zx_graph_to_db', importer), patch.object(ZXdb, 'export_graphdb_to_zx_graph', exporter):
            for name, spec in RULES.items():
                for case in spec['cases']:
                    options = case[2] if len(case) > 2 else {}
                    method = getattr(db, spec['db_method'])
                    if args.variant == 'full_reduce_optimized' and spec['db_method'] != 'hadamard_cancel':
                        from functools import partial
                        method = partial(db.apply_rule_optimized, spec['db_method'],
                                         exclusive_access=args.exclusive_access, incremental=args.incremental)
                    result = harness.run_case(db, name, case[0], case[1](),
                                              method, spec['pyzx'], **options)
                    record(result)
            for name, builder in build_suite(False):
                method = getattr(db, args.variant)
                if args.exclusive_access:
                    from functools import partial
                    method = partial(method, exclusive_access=True, incremental=args.incremental)
                result = harness.run_case(db, args.variant, name, builder(),
                                          method, pyzx_full_reduce, require_iso=False)
                record(result)
    finally:
        Path(args.output).write_text(json.dumps(dict(variant=args.variant,
            elapsed_s=time.monotonic()-started, results=results), indent=2))
        try:
            db.empty_graphdb(harness.GRAPH_ID)
        finally:
            db.close()
    if not harness.print_results(results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
