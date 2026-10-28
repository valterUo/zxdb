"""Exact graph snapshots for correctness checks after a timing campaign."""
from fractions import Fraction
import json
import multiprocessing
from pathlib import Path
import time

import pyzx as zx
from pyzx.graph.scalar import Scalar

from evaluation.workloads import digest, payload, metrics


def save_graph(directory, graph):
    """Save each distinct labeled graph once; call outside measured regions."""
    data = payload(graph)
    scalar = graph.scalar
    # PyZX's presentation serializer drops float factors within 1e-5 of one.
    data['scalar'] = dict(power2=scalar.power2, phase=str(scalar.phase),
                          floatfactor=str(scalar.floatfactor),
                          phasenodes=[str(p) for p in scalar.phasenodes],
                          sum_of_phases={str(p): c for p, c in scalar.sum_of_phases.items()},
                          is_zero=bool(scalar.is_zero), is_unknown=bool(scalar.is_unknown))
    key = digest(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + '.json')
    if path.exists():
        if digest(json.loads(path.read_text(encoding='utf-8'))) != key:
            raise ValueError(f'Corrupt graph evidence: {path}')
    else:
        path.write_text(json.dumps(data, sort_keys=True), encoding='utf-8')
    return key


def read_graph(directory, key):
    """Restore multiplicity, exact phases, boundary order and global scalar."""
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Expected a SHA-256 graph key')
    path = Path(directory) / (key + '.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if digest(data) != key:
        raise ValueError(f'Corrupt graph evidence: {path}')
    graph = zx.Graph(backend='multigraph')
    graph.set_auto_simplify(False)
    for vertex, kind, phase in data['vertices']:
        graph.add_vertex_indexed(vertex)
        graph.set_type(vertex, zx.VertexType(kind))
        graph.set_phase(vertex, Fraction(phase))
    for a, b, kind in data['edges']:
        graph.add_edge((a, b), zx.EdgeType(kind))
    graph.set_inputs(tuple(data['inputs']))
    graph.set_outputs(tuple(data['outputs']))
    graph.scalar = Scalar.from_json(data['scalar'])
    factor = data['scalar']['floatfactor']
    graph.scalar.floatfactor = complex(factor) if 'j' in factor else float(factor)
    return graph


def _ordered_audit_graph(graph):
    """Preserve a simple graph exactly, visiting boundary/leaf edges first.

    PyZX's gadget matcher can build large incident-edge lists before reaching
    a boundary that rejects a candidate. This ordering only affects the untimed
    audit. Multigraphs and self-loops retain their original representation.
    """
    edges = list(graph.edges())
    pairs = [tuple(sorted(graph.edge_st(e))) for e in edges]
    if len(set(pairs)) != len(pairs) or any(a == b for a, b in pairs):
        return graph
    result = zx.Graph()
    degree = {v: graph.vertex_degree(v) for v in graph.vertices()}
    for v in graph.vertices():
        result.add_vertex_indexed(v)
        result.set_type(v, graph.type(v))
        result.set_phase(v, graph.phase(v))
        result.set_ground(v, graph.is_ground(v))
    def priority(edge):
        a, b = graph.edge_st(edge)
        boundary = graph.type(a) == zx.VertexType.BOUNDARY or graph.type(b) == zx.VertexType.BOUNDARY
        return (not boundary, min(degree[a], degree[b]), min(a, b), max(a, b))
    for edge in sorted(edges, key=priority):
        result.add_edge(graph.edge_st(edge), graph.edge_type(edge))
    result.set_inputs(graph.inputs())
    result.set_outputs(graph.outputs())
    result.scalar = graph.scalar.copy()
    return result


def _fixed_point_worker(graph, directory, connection):
    """A structural audit, not a proof that the input and output are equivalent."""
    try:
        graph = _ordered_audit_graph(graph)
        before = payload(graph)
        before.pop('scalar')
        before_metrics = metrics(graph)
        start = time.perf_counter()
        zx.full_reduce(graph)
        seconds = time.perf_counter() - start
        after = payload(graph)
        after.pop('scalar')
        result = dict(status='completed', structure_changed=before != after,
                      before=before_metrics, after=metrics(graph), seconds=seconds)
        if directory is not None:
            result['output_key'] = save_graph(directory, graph)
        connection.send(result)
    except Exception as exc:
        connection.send(dict(status='inconclusive', reason=f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def check_reduction_fixed_point(graph, timeout_s=120, directory=None):
    """Run a bounded, untimed follow-up reduction on a separate graph copy.

    A structural change identifies work left for the reference implementation;
    unchanged structure is not a semantic certificate. Global scalar changes are
    excluded, consistently with the benchmark's projective equivalence contract.
    """
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_fixed_point_worker, args=(graph, directory, send))
    worker.start()
    send.close()
    try:
        if receive.poll(timeout_s):
            try:
                return receive.recv()
            except EOFError:
                return dict(status='inconclusive', reason='Fixed-point worker exited without a result')
        return dict(status='inconclusive', reason=f'Fixed-point audit exceeded {timeout_s:g}s')
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join()
        receive.close()
