"""Deterministic, explicitly parameterized workloads for research campaigns."""
import hashlib
import json
import random
from fractions import Fraction

import pyzx as zx

from evaluation import scalability as old


RULES = {
    'spider_fusion': ('spider_fusion', 'spider_simp'),
    'identity': ('remove_identities', 'id_simp'),
    'lcomp': ('local_complementation_rule', 'lcomp_simp'),
    'pivot': ('pivot_rule', 'pivot_simp'),
    'pivot_gadget': ('pivot_gadget_rule', 'pivot_gadget_simp'),
    'pivot_boundary': ('pivot_boundary_rule', 'pivot_boundary_simp'),
    'gadget_fusion': ('phase_gadget_fusion_rule', 'gadget_simp'),
    'bialgebra': ('bialgebra_simp', 'bialg_simp'),
    'full_reduce': ('full_reduce', 'full_reduce'),
}
FAMILIES = ('spider_fusion', 'identity', 'lcomp', 'pivot', 'pivot_gadget',
            'pivot_boundary', 'gadget_fusion', 'bialgebra', 'identity_chain',
            'random_clifford', 'random_clifford_t', 'random_t_heavy',
            'layered_local', 'layered_global')
FAMILIES += tuple('copies_' + rule for rule in RULES if rule != 'full_reduce')


def payload(g):
    """Exact labeled topology (including multiplicities); no layout dependence."""
    return {'vertices': [[int(v), int(g.type(v)), str(g.phase(v))]
                         for v in sorted(g.vertices())],
            'edges': sorted([min(g.edge_st(e)), max(g.edge_st(e)), int(g.edge_type(e))]
                            for e in g.edges()),
            'inputs': list(g.inputs()), 'outputs': list(g.outputs()),
            'scalar': g.scalar.to_json()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(spec):
    family, n, seed = spec['family'], spec['size'], spec['seed']
    if family.startswith('copies_'):
        motif = build(dict(spec, family=family[7:], size=2))
        # Linear disjoint union: tensor() repeatedly copies the whole prefix.
        g = zx.Graph()
        inputs, outputs = [], []
        for _ in range(n):
            ids = {v: g.add_vertex(motif.type(v), phase=motif.phase(v)) for v in motif.vertices()}
            for e in motif.edges():
                g.add_edge(tuple(ids[v] for v in motif.edge_st(e)), motif.edge_type(e))
            inputs.extend(ids[v] for v in motif.inputs())
            outputs.extend(ids[v] for v in motif.outputs())
            g.scalar.mult_with_scalar(motif.scalar)
        g.set_inputs(tuple(inputs)); g.set_outputs(tuple(outputs))
        return g
    if family.startswith('random_'):
        p_t = {'random_clifford': 0, 'random_clifford_t': .25,
               'random_t_heavy': .6}[family]
        return zx.generate.CNOT_HAD_PHASE_circuit(
            qubits=n, depth=spec['depth'], p_had=.2, p_t=p_t,
            clifford=family == 'random_clifford', seed=seed).to_graph()
    if family.startswith('layered_'):
        rng = random.Random(seed)
        c = zx.Circuit(n)
        for q in range(n):
            c.add_gate('HAD', q)
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if family == 'layered_global' or b == a + 1]
        count = spec.get('edges_per_layer', n - 1)
        if not 0 <= count <= len(pairs):
            raise ValueError('Requested edge count is infeasible for this locality')
        for _ in range(spec['depth']):
            for a, b in sorted(rng.sample(pairs, count)):
                c.add_gate('CNOT', a, b)
                c.add_gate('ZPhase', b, phase=Fraction(1, 4))
                c.add_gate('CNOT', a, b)
            for q in range(n):
                c.add_gate('XPhase', q, phase=Fraction(1, 2))
        return c.to_graph()
    if family == 'identity_chain':
        g = zx.Graph()
        bi = g.add_vertex(zx.VertexType.BOUNDARY)
        last = bi
        for _ in range(n):
            v = g.add_vertex(zx.VertexType.Z)
            g.add_edge((last, v)); last = v
        bo = g.add_vertex(zx.VertexType.BOUNDARY)
        g.add_edge((last, bo)); g.set_inputs((bi,)); g.set_outputs((bo,))
        return g
    g = getattr(old, 'g_' + family)(n)
    if family == 'gadget_fusion':
        outs = []
        for v in list(g.vertices())[:2]:
            b = g.add_vertex(zx.VertexType.BOUNDARY)
            g.add_edge((v, b)); outs.append(b)
        g.set_outputs(tuple(outs))
    # Expose scalar-only fan families as open maps; degree-one neighbours
    # otherwise prevent pivot-gadget/boundary matches in the historical scripts.
    if family in ('lcomp', 'pivot', 'pivot_gadget', 'pivot_boundary', 'bialgebra'):
        outputs = list(g.outputs())
        for v in list(g.vertices()):
            if g.type(v) != zx.VertexType.BOUNDARY and g.vertex_degree(v) == 1:
                b = g.add_vertex(zx.VertexType.BOUNDARY)
                g.add_edge((v, b)); outputs.append(b)
        g.set_outputs(tuple(outputs))
    return g


def rule_for(family):
    if family.startswith('copies_'):
        return family[7:]
    if family.startswith(('random_', 'layered_')):
        return 'full_reduce'
    return 'identity' if family == 'identity_chain' else family


def opportunities(spec):
    """Constructed witnesses, not a count of all possible/overlapping matches."""
    family, n = spec['family'], spec['size']
    if family.startswith(('random_', 'layered_')):
        return None
    if family.startswith('copies_'):
        return n * opportunities(dict(spec, family=family[7:], size=2))
    return n if family in ('identity', 'identity_chain', 'spider_fusion') else n-1 if family == 'gadget_fusion' else 1


def metrics(g):
    degrees = [g.vertex_degree(v) for v in g.vertices()]
    return dict(vertices=g.num_vertices(), edges=g.num_edges(),
                inputs=len(g.inputs()), outputs=len(g.outputs()),
                max_degree=max(degrees, default=0),
                mean_degree=sum(degrees) / max(1, len(degrees)),
                non_clifford=sum(g.type(v) != 0 and (g.phase(v) * 2) % 1 != 0
                                 for v in g.vertices()))


def make_plan(profile='smoke'):
    """Predeclared grids, optionally traversed with a timeout stopping rule."""
    if profile in ('adaptive', 'massive'):
        return adaptive_plan(profile)
    if profile not in ('smoke', 'study'):
        raise ValueError('Unknown profile')
    broad = profile == 'study'
    cases = []
    for family in FAMILIES:
        circuit = family.startswith(('random_', 'layered_'))
        sizes = ([2, 4, 8, 16] if circuit else [2, 8, 32, 128]) if broad else [2]
        for size in sizes:
            depths = ([2, 4] if family.startswith('layered_') else [20, 80]) if circuit else [None]
            if not broad:
                depths = [2 if family.startswith('layered_') else 8] if circuit else [None]
            for depth in depths:
                for seed in range(5 if broad and circuit else 1):
                    spec = dict(family=family, size=size, depth=depth, seed=seed,
                                scale='qubits' if circuit else 'patterns' if family in
                                ('identity', 'spider_fusion', 'identity_chain') or family.startswith('copies_') else
                                'gadgets' if family == 'gadget_fusion' else 'fan_size')
                    spec['id'] = digest(spec)[:16]
                    cases.append(spec)
    plan = dict(schema=1, profile=profile, repetitions=5 if broad else 2,
                warmups=1, order_seed=20260908, block_timeout_s=180,
                query_timeout_s=30, max_vertices=100000, cases=cases)
    plan['baseline_contract'] = 'pyzx-0.10.3-native-v2'
    return plan


def adaptive_plan(profile='adaptive'):
    cases = []
    for family in FAMILIES:
        circuit = family.startswith(('random_', 'layered_'))
        independent = family.startswith('copies_') or family in ('identity', 'identity_chain', 'spider_fusion')
        maximum_exponent = 16 if circuit else 18 if independent else 10
        # Depth is the growing axis for circuits; width stays controlled.
        for width in ([2, 8, 32] if circuit else [None]):
            for exponent in range(2 if circuit else 1, maximum_exponent):
                for seed in range(3 if circuit else 1):
                    spec = dict(family=family, size=width if circuit else 2**exponent,
                                depth=2**exponent if circuit else None, seed=seed,
                                scale='qubits' if circuit else 'patterns' if family.startswith('copies_') or
                                family in ('identity', 'identity_chain', 'spider_fusion') else
                                'gadgets' if family == 'gadget_fusion' else 'fan_size')
                    spec['id'] = digest(spec)[:16]; cases.append(spec)
    return dict(schema=1, profile=profile, baseline_contract='pyzx-0.10.3-native-v2',
                repetitions=2, warmups=1, order_seed=20260908,
                block_timeout_s=120, engine_timeout_s=15, pyzx_timeout_s=15,
                pyzx_max_rss_mb=4096, query_timeout_s=10,
                max_vertices=2000000, max_edges=4000000, applicability_probe='motif',
                adaptive=dict(consecutive_timeouts=3, stop_on='both'), cases=cases)
