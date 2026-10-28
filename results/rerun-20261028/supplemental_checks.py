"""One-off exact post-run checks, with source embedded in the saved evidence."""
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from evaluation.cases import copy_two_leaves
from evaluation.circuits import build_suite
from evaluation.core import certify_unitary_rewrite
from evaluation.evidence import read_graph, save_graph, _ordered_audit_graph


def exact_zero_tensor(g):
    """Exhaustive symbolic contraction for the tiny zero-map fixture only.

    Assign a bit to each edge endpoint. Simple edges impose equality;
    Hadamard edges contribute (-1)**(a*b)/sqrt(2). Z and X spiders use
    their exact computational-basis definitions. Global scalar is irrelevant
    when every amplitude of the unscaled tensor is exactly zero.
    """
    import sympy as s
    edges = list(g.edges())
    assert len(edges) <= 3 and len(g.inputs()) == 1 and not g.outputs()
    legs = {v: [] for v in g.vertices()}
    for i, e in enumerate(edges):
        a, b = g.edge_st(e)
        legs[a].append(2*i)
        legs[b].append(2*i+1)
    amplitudes = []
    for external in (0, 1):
        total = s.Integer(0)
        for bits in itertools.product((0, 1), repeat=2*len(edges)):
            term = s.Integer(1)
            for i, e in enumerate(edges):
                a, b = bits[2*i:2*i+2]
                kind = int(g.edge_type(e))
                assert kind in (1, 2)
                term *= int(a == b) if kind == 1 else (-1)**(a*b)/s.sqrt(2)
            for v, indices in legs.items():
                values = [bits[i] for i in indices]
                kind = int(g.type(v))
                if kind == 0:
                    assert v == g.inputs()[0] and len(values) == 1
                    term *= int(values[0] == external)
                    continue
                phase = s.exp(s.I*s.pi*s.Rational(str(g.phase(v))))
                if kind == 1:
                    term *= int(all(x == 0 for x in values)) + phase*int(all(x == 1 for x in values))
                elif kind == 2:
                    term *= (1 + phase*(-1)**sum(values))/s.sqrt(2)**len(values)
                else:
                    raise ValueError('Unsupported vertex')
            total += term
        amplitudes.append(s.simplify(s.expand_complex(total)))
    return {'status': 'verified' if all(x == 0 for x in amplitudes) else 'failed',
            'reason': 'Exact symbolic zero tensor', 'amplitudes': list(map(str, amplitudes))}


def main():
    root = Path(sys.argv[1])
    assert 'Run state: finished.' in (root/'REPORT.md').read_text()
    rows = [json.loads(x) for x in (root/'observations.jsonl').read_text().splitlines()]
    unresolved = [r for r in rows if r['stage'] == 'correctness' and r['status'] == 'inconclusive']
    assert {r['case'] for r in unresolved} == {'two_leaves', 'wide_q10_d32_s1077', 'wide_q12_d36_s1078'}
    builders = dict(build_suite())
    builders['two_leaves'] = copy_two_leaves
    report = {'audit_source': Path(__file__).read_text(), 'checks': []}
    for row in unresolved:
        original = builders[row['case']]()
        assert save_graph(root/'graphs', original) == row['graph_evidence']['original']
        checks = {}
        for engine, key in row['graph_evidence'].items():
            result = _ordered_audit_graph(read_graph(root/'graphs', key))
            assert save_graph(root/'graphs', result) == key
            if row['case'] == 'two_leaves':
                checks[engine] = exact_zero_tensor(result)
            elif engine != 'original':
                checks[engine] = certify_unitary_rewrite(original, result, 120)
        report['checks'].append({'case': row['case'], 'variant': row['variant'], 'graph_evidence': row['graph_evidence'], 'checks': checks})
        (root/'supplemental-correctness.json').write_text(json.dumps(report, indent=2))
        print(row['case'], checks, flush=True)


if __name__ == '__main__':
    main()
