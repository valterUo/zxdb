"""Deterministic, explicitly parameterized workloads for research campaigns."""
import hashlib
import json
import random
from fractions import Fraction

import pyzx as zx

from evaluation import patterns as old


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
    'hadamard_cancel': ('hadamard_cancel', 'id_simp'),
    'supplementarity': ('supplementarity_simp', 'supplementarity_simp'),
    'copy': ('copy_simp', 'copy_simp'),
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
        counts = spec.get('motif_counts', {'2': n})
        if sum(counts.values()) != n or any(int(k) not in (2, 3) or not isinstance(v, int) or v < 0 for k, v in counts.items()):
            raise ValueError('Invalid motif mixture')
        motifs = {int(k): build(dict(spec, family=family[7:], size=int(k))) for k, count in counts.items() if count}
        sizes = [int(k) for k, count in sorted(counts.items()) for _ in range(count)]
        if 'motif_counts' in spec:
            random.Random(seed).shuffle(sizes)
        # Linear disjoint union: tensor() repeatedly copies the whole prefix.
        g = zx.Graph()
        inputs, outputs = [], []
        for motif_size in sizes:
            motif = motifs[motif_size]
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
    g = getattr(old, 'g_' + ('hadamard' if family == 'hadamard_cancel' else family))(n)
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
        return sum(count * opportunities(dict(spec, family=family[7:], size=int(size)))
                   for size, count in spec.get('motif_counts', {'2': n}).items())
    return n if family in ('identity', 'identity_chain', 'spider_fusion') else n-1 if family == 'gadget_fusion' else 1


def metrics(g):
    degrees = [g.vertex_degree(v) for v in g.vertices()]
    return dict(vertices=g.num_vertices(), edges=g.num_edges(),
                inputs=len(g.inputs()), outputs=len(g.outputs()),
                max_degree=max(degrees, default=0),
                mean_degree=sum(degrees) / max(1, len(degrees)),
                non_clifford=sum(g.type(v) != 0 and (g.phase(v) * 2) % 1 != 0
                                 for v in g.vertices()))


