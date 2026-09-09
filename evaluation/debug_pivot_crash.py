"""Opt-in crash investigation against a disposable server (default port 7688)."""
import argparse
import json
from pathlib import Path
from fractions import Fraction
import pyzx as zx
from neo4j import GraphDatabase
from evaluation.research_core import load, read
from evaluation.research_workloads import payload
from zxdb.zxdb import ZXdb


def restore(data):
    g = zx.Graph(backend='multigraph'); g.set_auto_simplify(False)
    ids = {v: g.add_vertex(zx.VertexType(t), phase=Fraction(p)) for v, t, p in data['vertices']}
    for a, b, t in data['edges']:
        g.add_edge((ids[a], ids[b]), zx.EdgeType(t))
    g.set_inputs(tuple(ids[v] for v in data['inputs']))
    g.set_outputs(tuple(ids[v] for v in data['outputs']))
    return g


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', default='results-study/inputs/49324f2631b8374c.json')
    parser.add_argument('--output', default='evaluation/audit_results/pivot_crash')
    parser.add_argument('--uri', default='bolt://localhost:7688')
    parser.add_argument('--query', help='Execute this Cypher file once instead of full_reduce')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--edge-scan', action='store_true', help='Disposable server only: remove node indexes to force the edge scan')
    parser.add_argument('--no-snapshots', action='store_true')
    args = parser.parse_args()
    directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
    gid = 'pivot_debug'
    with GraphDatabase.driver(args.uri, auth=('', '')) as driver:
        with driver.session() as s:
            if s.run('MATCH(n) WHERE n.graph_id IS NULL OR n.graph_id <> $gid RETURN count(n) AS n', gid=gid).single()['n']:
                raise RuntimeError('Debug server contains unrelated data')
    db = ZXdb(uri=args.uri)
    if args.edge_scan:
        with db.driver.session() as s:
            for statement in ['DROP INDEX ON :Node(t)', 'DROP INDEX ON :Node(graph_id)']:
                s.run(statement).consume()
    input_data = json.loads(Path(args.input).read_text())
    original = db._run_count
    counter = 0
    def trace(tx, query, key, graph_id):
        nonlocal counter
        counter += 1
        if not args.no_snapshots:
            (directory / 'before.json').write_text(json.dumps(payload(read(db, graph_id)), indent=2))
        (directory / 'query.cypher').write_text(query)
        print(counter, query.splitlines()[0], flush=True)
        result = original(tx, query, key, graph_id)
        print('count', result, flush=True)
        return result
    db._run_count = trace
    for iteration in range(args.repeat):
        print('iteration', iteration, flush=True)
        load(db, restore(input_data), gid)
        if args.query:
            with db.driver.session() as s:
                print(s.run(Path(args.query).read_text(), graph_id=gid).data())
        else:
            db.full_reduce(gid)
    (directory / 'after.json').write_text(json.dumps(payload(read(db, gid)), indent=2))
    db.empty_graphdb(gid)
    db.close()


if __name__ == '__main__':
    main()
