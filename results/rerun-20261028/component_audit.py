"""Untimed exact component-wise comparison against each recorded PyZX output.

Preserves colors, phases, wire multiplicities, and ordered boundary labels.
Scalar remains excluded, as in the original structural comparison.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path.cwd()))
import networkx as nx
from evaluation.evidence import read_graph


def graph(g):
    h = nx.MultiGraph()
    roles = {v: ('input', i) for i, v in enumerate(g.inputs())}
    roles.update({v: ('output', i) for i, v in enumerate(g.outputs())})
    for v in g.vertices():
        h.add_node(v, label=(int(g.type(v)), str(Fraction(g.phase(v)) % 2),
                             roles.get(v, ('internal', 0))))
    for e in g.edges():
        h.add_edge(*g.edge_st(e), t=int(g.edge_type(e)))
    return h


def signature(g):
    return (g.number_of_nodes(), g.number_of_edges(),
            tuple(sorted((d['label'], g.degree(v)) for v, d in g.nodes(data=True))),
            tuple(sorted(Counter(d['t'] for _, _, d in g.edges(data=True)).items())))


def compare(a, b, timeout=10):
    start = time.monotonic()
    deadline = start+timeout
    groups = []
    for h in (graph(a), graph(b)):
        grouped = defaultdict(list)
        for vertices in nx.connected_components(h):
            component = h.subgraph(vertices).copy()
            grouped[signature(component)].append(component)
        groups.append(grouped)
    left, right = groups
    if {k:len(v) for k,v in left.items()} != {k:len(v) for k,v in right.items()}:
        return dict(status='different', reason='Different labeled connected-component invariants')
    class Matcher(nx.algorithms.isomorphism.MultiGraphMatcher):
        def candidate_pairs_iter(self):
            for pair in super().candidate_pairs_iter():
                if time.monotonic() > deadline:
                    raise TimeoutError('Component isomorphism exceeded time bound')
                yield pair
    try:
        for key, components in left.items():
            candidates = list(right[key])
            for component in components:
                for index, candidate in enumerate(candidates):
                    matcher = Matcher(component, candidate,
                                      node_match=lambda x,y:x['label']==y['label'],
                                      edge_match=lambda x,y:Counter(d['t'] for d in x.values()) == Counter(d['t'] for d in y.values()))
                    try:
                        equal = matcher.is_isomorphic()
                    finally:
                        matcher.reset_recursion_limit()
                    if equal:
                        candidates.pop(index)
                        break
                else:
                    return dict(status='different', reason='No exact component match')
        return dict(status='equal', reason='Exact labeled component-wise multigraph isomorphism',
                    components=sum(map(len,left.values())), elapsed_s=time.monotonic()-start)
    except (TimeoutError, RecursionError) as exc:
        return dict(status='inconclusive', reason=str(exc), elapsed_s=time.monotonic()-start)


def main():
    # Counterexamples ensure that equal degree sequences cannot pass this audit
    # after phase, boundary-order or wire-type/multiplicity changes.
    import pyzx as zx
    g = zx.Graph()
    a = g.add_vertex(zx.VertexType.BOUNDARY)
    b = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
    c = g.add_vertex(zx.VertexType.BOUNDARY)
    g.add_edges([(a,b),(b,c)])
    g.set_inputs((a,)); g.set_outputs((c,))
    assert compare(g,g.copy())['status']=='equal'
    changed=g.copy(); changed.set_phase(b,Fraction(3,4))
    assert compare(g,changed)['status']=='different'
    wires=zx.Graph()
    vertices=[wires.add_vertex(zx.VertexType.BOUNDARY) for _ in range(4)]
    wires.add_edges([(vertices[0],vertices[2]),(vertices[1],vertices[3])])
    wires.set_inputs(tuple(vertices[:2])); wires.set_outputs(tuple(vertices[2:]))
    changed=wires.copy(); changed.set_outputs(tuple(reversed(changed.outputs())))
    assert compare(wires,changed)['status']=='different'
    multi=zx.Graph(backend='multigraph'); multi.set_auto_simplify(False)
    a=multi.add_vertex(zx.VertexType.Z); b=multi.add_vertex(zx.VertexType.Z)
    multi.add_edge((a,b),zx.EdgeType.SIMPLE); multi.add_edge((a,b),zx.EdgeType.HADAMARD)
    changed=zx.Graph(backend='multigraph'); changed.set_auto_simplify(False)
    c=changed.add_vertex(zx.VertexType.Z); d=changed.add_vertex(zx.VertexType.Z)
    changed.add_edge((c,d),zx.EdgeType.SIMPLE); changed.add_edge((c,d),zx.EdgeType.SIMPLE)
    assert compare(multi,changed)['status']=='different'
    root = Path(__file__).resolve().parent/'performance'
    rows = [json.loads(x) for x in (root/'observations.jsonl').read_text().splitlines()]
    refs = {(r['case'],r['repetition']):r for r in rows if r['engine']=='pyzx' and 'graphs' in r}
    selected = {}
    for row in rows:
        if row['engine'] not in ('cypher','hybrid') or row['warmup'] or 'graphs' not in row:
            continue
        if row.get('against_pyzx',{}).get('graph',{}).get('status') != 'inconclusive':
            continue
        ref = refs[(row['case'],row['repetition'])]
        selected.setdefault((row['graphs']['result'],ref['graphs']['result']), []).append(row)
    report = dict(method=__doc__, counterexample_self_checks='passed', audit_source=Path(__file__).read_text(), checks=[])
    for (dbkey, refkey), trials in selected.items():
        result = compare(read_graph(root/'graphs',dbkey), read_graph(root/'graphs',refkey))
        report['checks'].append(dict(db_hash=dbkey, pyzx_hash=refkey, result=result,
                                    trials=[dict(case=r['case'],engine=r['engine'],repetition=r['repetition'],spec=r['spec']) for r in trials]))
        (root/'component-isomorphism.json').write_text(json.dumps(report,indent=2))
        print(len(report['checks']), '/', len(selected), trials[0]['spec'], result, flush=True)


if __name__=='__main__':
    main()
