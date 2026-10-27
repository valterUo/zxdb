"""Measurement and correctness primitives. No global database deletion."""
import time
from collections import Counter
from fractions import Fraction

import numpy as np
import pyzx as zx
from neo4j import Query

from evaluation.workloads import RULES, payload


# The planner can otherwise prefer graph_id (every vertex of this diagram)
# over the selective vertex ID, especially with cached plans on empty graphs.
LOAD_EDGES_QUERY = ('USING INDEX :Node(id) UNWIND $es AS e '
                    'MATCH(a:Node {graph_id:$gid,id:e.a}),'
                    '(b:Node {graph_id:$gid,id:e.b}) '
                    'CREATE(a)-[:Wire {t:e.t,graph_id:$gid}]->(b)')


class SessionProxy:
    def __init__(self, session, counter, managed=False):
        self.session, self.counter = session, counter
        self.managed = managed

    def __enter__(self):
        self.session.__enter__()
        return self

    def __exit__(self, *args):
        return self.session.__exit__(*args)

    def run(self, query, *args, **kwargs):
        self.counter.submissions += 1
        self.counter.last_query = str(query)
        remaining = self.counter.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Block deadline exceeded')
        statement = str(query) if self.managed else Query(str(query), timeout=min(remaining, self.counter.timeout))
        return self.session.run(statement,
                                *args, **kwargs)

    def execute_write(self, callback, *args, **kwargs):
        return self.session.execute_write(
            lambda tx: callback(SessionProxy(tx, self.counter, managed=True), *args, **kwargs))

    def execute_read(self, callback, *args, **kwargs):
        return self.session.execute_read(
            lambda tx: callback(SessionProxy(tx, self.counter, managed=True), *args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.session, name)


class CountingDriver:
    """Counts Cypher submissions, not network packets or internal operators."""
    def __init__(self, driver, timeout=30, deadline=float('inf')):
        self.driver, self.timeout, self.deadline = driver, timeout, deadline
        self.submissions = 0
        self.last_query = None

    def session(self, *args, **kwargs):
        return SessionProxy(self.driver.session(*args, **kwargs), self)

    def __getattr__(self, name):
        return getattr(self.driver, name)


def load(db, g, gid):
    input_indices = {v: i for i, v in enumerate(g.inputs())}
    output_indices = {v: i for i, v in enumerate(g.outputs())}
    with db.driver.session() as s:
        s.run('MATCH(n:Node {graph_id:$gid}) DETACH DELETE n', gid=gid).consume()
        vs = [dict(id=int(v), t=int(g.type(v)), phase=float(g.phase(v)),
                   is_input=v in input_indices, is_output=v in output_indices,
                   io_index=output_indices.get(v, input_indices.get(v))) for v in g.vertices()]
        # Encode an explicit zero scalar as a disconnected pi spider. This
        # keeps zero observable without adding a separate scalar-storage API.
        if (g.scalar.is_zero or g.scalar.floatfactor == 0
                or any(p % 2 == 1 for p in g.scalar.phasenodes)):
            vs.append(dict(id=max(g.vertices(), default=-1) + 1, t=1, phase=1.0,
                           is_input=False, is_output=False, io_index=None))
        es = [dict(a=int(g.edge_s(e)), b=int(g.edge_t(e)), t=int(g.edge_type(e))) for e in g.edges()]
        for start in range(0, len(vs), 1000):
            s.run('UNWIND $vs AS v CREATE(n:Node) '
                  'SET n.id=v.id,n.t=v.t,n.phase=v.phase,n.graph_id=$gid,n.io_index=v.io_index '
                  'FOREACH(_ IN CASE WHEN v.is_input THEN [1] ELSE [] END | SET n:Input) '
                  'FOREACH(_ IN CASE WHEN v.is_output THEN [1] ELSE [] END | SET n:Output)',
                  vs=vs[start:start+1000], gid=gid).consume()
        for start in range(0, len(es), 1000):
            s.run(LOAD_EDGES_QUERY,
                  es=es[start:start+1000], gid=gid).consume()


def read(db, gid):
    g = zx.Graph(backend='multigraph'); g.set_auto_simplify(False)
    with db.driver.session() as s:
        nodes = s.run('MATCH(n:Node {graph_id:$gid}) RETURN id(n) AS id,n.t AS t,'
                      'n.phase AS phase,labels(n) AS labels,n.io_index AS ix ORDER BY id', gid=gid).data()
        edges = s.run('MATCH(a:Node {graph_id:$gid})-[e:Wire]->(b:Node {graph_id:$gid}) '
                      'RETURN id(a) AS a,id(b) AS b,e.t AS t', gid=gid).data()
    ids = {r['id']: g.add_vertex(zx.VertexType(r['t']),
                                phase=Fraction(r['phase'] or 0).limit_denominator(10**6)) for r in nodes}
    for r in edges:
        g.add_edge((ids[r['a']], ids[r['b']]), zx.EdgeType(r['t']))
    for label, setter in [('Input', g.set_inputs), ('Output', g.set_outputs)]:
        io = [r for r in nodes if label in r['labels']]
        if any(r['ix'] is None for r in io) or len({r['ix'] for r in io}) != len(io):
            raise ValueError('Missing or duplicate boundary ordering')
        setter(tuple(ids[r['id']] for r in sorted(io, key=lambda r: r['ix'])))
    return g


def validate_graph(g):
    boundaries = {v for v in g.vertices() if g.type(v) == zx.VertexType.BOUNDARY}
    io = list(g.inputs()) + list(g.outputs())
    if set(io) != boundaries or len(io) != len(set(io)):
        raise ValueError('Every boundary must have exactly one ordered I/O role')
    if any(g.vertex_degree(v) != 1 for v in boundaries):
        raise ValueError('Boundary degree must equal one')
    if any(g.type(v) not in (0, 1, 2, 3) for v in g.vertices()):
        raise OverflowError('Unsupported vertex type for tensor oracle')
    if any(g.type(v) == 3 and g.phase(v) % 2 != 1 for v in g.vertices()):
        raise OverflowError('Only pi-phase H-boxes supported by tensor oracle')


def tensor(g):
    h = g.copy()
    # Projective semantics: discard only provably nonzero global factors.
    # sqrt(2)**power2 can underflow even for a perfectly ordinary nonzero map.
    scalar = h.scalar
    if scalar.is_unknown:
        raise OverflowError('Unknown scalar')
    if not np.isfinite(scalar.floatfactor):
        raise OverflowError('Nonfinite scalar factor')
    if scalar.is_zero or scalar.floatfactor == 0 or any(p % 2 == 1 for p in scalar.phasenodes):
        scalar.is_zero = True
    else:
        scalar.power2 = 0
        scalar.phase = Fraction(0)
        scalar.floatfactor = 1.0
        if scalar.sum_of_phases:
            value = sum(c * np.exp(1j * np.pi * float(p)) for p, c in scalar.sum_of_phases.items())
            if abs(value) < 1e-12:
                raise OverflowError('Scalar sum is numerically ambiguous')
        if abs(scalar.to_number()) < 1e-12:
            raise OverflowError('Scalar product is numerically ambiguous')
    for i, v in enumerate(h.vertices()):
        h.set_row(v, 0 if v in h.inputs() else h.num_vertices()+1 if v in h.outputs() else i+1)
    from evaluation.oracle import _quimb_tensor
    array = _quimb_tensor(h)
    if array is None:
        raise OverflowError('Tensor resource bound exceeded or unsupported contraction')
    return array.ravel()


def proven_zero(g):
    """Exact zero factors we can recognize without a floating-point threshold."""
    scalar = g.scalar
    return (scalar.is_zero or scalar.floatfactor == 0
            or any(p % 2 == 1 for p in scalar.phasenodes)
            or any(g.type(v) in (1, 2) and g.vertex_degree(v) == 0
                   and g.phase(v) % 2 == 1 for v in g.vertices()))


def verify(original, result):
    try:
        validate_graph(original); validate_graph(result)
        if (len(original.inputs()), len(original.outputs())) != (len(result.inputs()), len(result.outputs())):
            return dict(status='failed', reason='I/O arity changed')
        za, zb = proven_zero(original), proven_zero(result)
        if za and zb:
            return dict(status='verified', reason='Explicit zero factors on both graphs')
        a, b = tensor(original), tensor(result)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return dict(status='inconclusive', reason='Nonfinite tensor contraction')
        ma, mb = np.max(np.abs(a), initial=0), np.max(np.abs(b), initial=0)
        if (za and mb > 1e-8) or (zb and ma > 1e-8):
            return dict(status='failed', reason='Proven zero map versus nonzero tensor')
        if za or zb or min(ma, mb) < 1e-8:
            return dict(status='inconclusive', reason='Near-zero tensor without a matching exact zero proof')
        # Scale before the norm to avoid overflow when squaring large amplitudes.
        a, b = a/ma, b/mb
        a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
        overlap = np.vdot(a, b)
        error = float(np.linalg.norm(b - overlap*a))
        return dict(status='verified' if error < 1e-8 else 'failed',
                    reason='Ordered tensor equality up to nonzero global scalar', residual=error)
    except OverflowError as exc:
        return dict(status='inconclusive', reason=str(exc))
    except Exception as exc:
        return dict(status='failed', reason=f'{type(exc).__name__}: {exc}')


def _identity_certificate_worker(original, result, connection):
    """Called in a bounded subprocess; the caller supplies a known unitary input."""
    try:
        validate_graph(original); validate_graph(result)
        q = len(original.inputs())
        if len(original.outputs()) != q or (len(result.inputs()), len(result.outputs())) != (q, q):
            connection.send(dict(status='failed', reason='Circuit I/O arity changed'))
            return
        if proven_zero(result):
            connection.send(dict(status='failed', reason='Unitary circuit reduced to an explicit zero map'))
            return
        composed = original.adjoint()
        composed.compose(result)
        zx.full_reduce(composed)
        validate_graph(composed)
        scalar = composed.scalar
        nonzero = (not scalar.is_zero and not scalar.is_unknown
                   and np.isfinite(scalar.floatfactor) and scalar.floatfactor != 0
                   and not scalar.sum_of_phases
                   and all(p % 2 != 1 for p in scalar.phasenodes))
        identity = (composed.num_vertices() == 2*q and composed.num_edges() == q
                    and all(composed.edge_type(e) == zx.EdgeType.SIMPLE for e in composed.edges())
                    and all(composed.connected(a, b) for a, b in zip(composed.inputs(), composed.outputs())))
        connection.send(dict(status='verified' if identity and nonzero else 'inconclusive',
                             reason='PyZX identity certificate for a known unitary circuit' if identity and nonzero
                             else 'Composition did not reduce to a certified nonzero ordered identity',
                             evidence='algebraic_identity', scalar=scalar.to_json()))
    except Exception as exc:
        connection.send(dict(status='inconclusive', reason=f'Identity certificate: {type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def certify_unitary_rewrite(original, result, timeout_s=120):
    """Prove a rewrite of a KNOWN UNITARY circuit via A-dagger B = c I, c != 0.

    This is not valid for arbitrary ZX maps: callers must establish unitarity
    from circuit construction. It is deliberately not a fallback in verify().
    The child process bounds expensive PyZX proof searches on Windows too.
    """
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    worker = context.Process(target=_identity_certificate_worker, args=(original, result, send))
    worker.start()
    send.close()
    try:
        if receive.poll(timeout_s):
            try:
                return receive.recv()
            except EOFError:
                return dict(status='inconclusive', reason='Identity certificate worker exited without a result')
        return dict(status='inconclusive', reason=f'Identity certificate exceeded {timeout_s:g}s')
    finally:
        if worker.is_alive(): worker.terminate()
        worker.join()
        receive.close()


def compare_outputs(db_graph, pyzx_graph, max_iso_vertices=256, iso_timeout_s=1.0):
    """Independent evidence levels; structural disagreement is not a semantic failure.

    ZX normal forms need not be unique. Isomorphism preserves ordered boundaries,
    spider colors/phases modulo 2, wire types, parallel edges and self-loops.
    Scalars are intentionally excluded from the structural comparison.
    """
    import networkx as nx
    start = time.perf_counter()
    tensor_check = verify(pyzx_graph, db_graph)
    def degrees(g):
        return Counter(g.vertex_degree(v) for v in g.vertices())
    da, db = degrees(db_graph), degrees(pyzx_graph)
    degree_check = dict(status='equal' if da == db else 'different',
                        db_histogram=sorted(da.items()), pyzx_histogram=sorted(db.items()))
    def graph(g):
        h = nx.MultiGraph()
        roles = {v: ('input', i) for i, v in enumerate(g.inputs())}
        roles.update({v: ('output', i) for i, v in enumerate(g.outputs())})
        for v in g.vertices():
            h.add_node(v, label=(int(g.type(v)), str(Fraction(g.phase(v)).limit_denominator(10**6) % 2),
                                 roles.get(v, ('internal', 0))))
        for e in g.edges():
            h.add_edge(*g.edge_st(e), t=int(g.edge_type(e)))
        return h
    if da != db or db_graph.num_edges() != pyzx_graph.num_edges():
        graph_check = dict(status='different', reason='Different degree sequence or edge count')
    elif max(db_graph.num_vertices(), pyzx_graph.num_vertices()) > max_iso_vertices or max(
            db_graph.num_edges(), pyzx_graph.num_edges()) > 4096:
        graph_check = dict(status='inconclusive', reason='Exact isomorphism size bound exceeded')
    else:
        a, b = graph(db_graph), graph(pyzx_graph)
        deadline = time.monotonic() + iso_timeout_s
        class BoundedMatcher(nx.algorithms.isomorphism.MultiGraphMatcher):
            def candidate_pairs_iter(self):
                for pair in super().candidate_pairs_iter():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Exact isomorphism time bound exceeded')
                    yield pair
        try:
            # categorical_multiedge_match uses sets, losing type multiplicity.
            matcher = BoundedMatcher(a, b, node_match=lambda x, y: x['label'] == y['label'],
                                     edge_match=lambda x, y: Counter(e['t'] for e in x.values()) ==
                                     Counter(e['t'] for e in y.values()))
            try:
                equal = matcher.is_isomorphic()
            finally:
                matcher.reset_recursion_limit()
            graph_check = dict(status='equal' if equal else 'different', reason='Exact labeled multigraph isomorphism')
        except (TimeoutError, RecursionError) as exc:
            graph_check = dict(status='inconclusive', reason=str(exc))
    return dict(tensor=tensor_check, graph=graph_check, degree_sequence=degree_check,
                comparison_s=time.perf_counter()-start)


def reference(g, rule, deadline):
    fn = getattr(zx.simplify, RULES[rule][1])
    # PyZX 0.10.3: these three APIs apply one matching batch. The others
    # already iterate internally; full_reduce is the public algorithm itself.
    batch_only = rule in ('pivot_gadget', 'pivot_boundary', 'gadget_fusion')
    for _ in range(1000):
        if time.monotonic() > deadline:
            raise TimeoutError('Reference deadline exceeded')
        changed = fn(g)
        if not batch_only or not changed:
            return
    raise RuntimeError('Reference did not converge in 1000 passes')
