"""Reproduce audit findings on a DISPOSABLE Memgraph database.

Refuses to start unless the database is empty. Uses direct fixture loading to
avoid the importer masking defects. Run: python -m evaluation.audit_system_probes
"""
import json
from fractions import Fraction
from pathlib import Path

import pyzx as zx
import numpy as np
from evaluation.harness import _clean_copy, _quimb_equal, _quimb_tensor
from zxdb.zxdb import ZXdb

OUT = Path('tmp/audit')


def load(db, g, gid='audit'):
    vertices = [{'id': int(v), 't': int(g.type(v)), 'phase': float(g.phase(v))}
                for v in g.vertices()]
    edges = [{'a': int(g.edge_s(e)), 'b': int(g.edge_t(e)), 't': int(g.edge_type(e))}
             for e in g.edges()]
    with db.driver.session() as s:
        s.run('UNWIND $vs AS v CREATE (n:Node) SET n = v, n.graph_id = $gid',
              vs=vertices, gid=gid).consume()
        s.run('UNWIND $es AS e MATCH (a:Node {graph_id:$gid,id:e.a}), '
              '(b:Node {graph_id:$gid,id:e.b}) '
              'CREATE (a)-[:Wire {t:e.t,graph_id:$gid}]->(b)', es=edges, gid=gid).consume()
        for label, vs in [('Input', g.inputs()), ('Output', g.outputs())]:
            s.run('UNWIND $vs AS v MATCH (n:Node {graph_id:$gid,id:v.id}) '
                  f'SET n:{label}, n.io_index=v.idx',
                  vs=[{'id': int(v), 'idx': i} for i, v in enumerate(vs)], gid=gid).consume()


def wire():
    g = zx.Graph()
    a, b = [g.add_vertex(zx.VertexType.BOUNDARY) for _ in range(2)]
    g.add_edge((a, b))
    g.set_inputs((a,))
    g.set_outputs((b,))
    return g


def export(db, gid='audit'):
    return db.export_graphdb_to_zx_graph(gid, str(OUT / 'export.json'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    with ZXdb() as db:
        if db.node_count():
            raise RuntimeError('Audit probes require an empty disposable database')
        try:
            g = wire()
            source = OUT / 'input.json'
            source.write_text(g.to_json(), encoding='utf-8')
            db.import_zx_graph_json_to_graphdb(str(source), 'audit', save_metadata=False)
            results['default_import_nodes'] = db.node_count()
            load(db, wire(), 'other')
            db.import_zx_graph_json_to_graphdb(str(source), 'audit', save_metadata=False,
                                             initialize_empty=True)
            results['other_graph_after_import'] = export(db, 'other').num_vertices()
            db.empty_graphdb('audit')

            from evaluation.cases import sf_chain3
            load(db, sf_chain3(), 'other')
            results['other_before_missing_graph_reduce'] = export(db, 'other').num_vertices()
            db.full_reduce('nonexistent')
            results['other_after_missing_graph_reduce'] = export(db, 'other').num_vertices()
            db.empty_graphdb('other')

            g = wire()
            g.add_vertex(zx.VertexType.Z, phase=1)
            load(db, g)
            db.full_reduce('audit')
            reduced = export(db)
            results['zero_component_harness_equivalent'] = _quimb_equal(g, reduced)
            results['zero_component_before_tensor'] = str(_quimb_tensor(g))
            results['zero_component_after_tensor'] = str(_quimb_tensor(reduced))
            results['zero_component_exact_input_is_zero'] = bool(np.allclose(_clean_copy(g).to_tensor(), 0))
            results['zero_component_exact_output_is_zero'] = bool(np.allclose(_clean_copy(reduced).to_tensor(), 0))
            db.empty_graphdb('audit')

            g = wire()
            g.scalar.is_zero = True
            source.write_text(g.to_json(), encoding='utf-8')
            db.import_zx_graph_json_to_graphdb(str(source), 'audit', save_metadata=False,
                                             initialize_empty=True)
            reduced = export(db)
            results['input_zero_scalar'] = g.scalar.is_zero
            results['export_zero_scalar'] = reduced.scalar.is_zero
            results['harness_zero_scalar_equivalent'] = _quimb_equal(g, reduced)
            db.empty_graphdb('audit')

            # Color-dual fusion creates two H wires between Z and X. These
            # collapse to one H wire; Hopf must not cancel them as simple wires.
            g = zx.Graph()
            bi = g.add_vertex(zx.VertexType.BOUNDARY)
            bo = g.add_vertex(zx.VertexType.BOUNDARY)
            a = g.add_vertex(zx.VertexType.Z, phase=Fraction(1, 4))
            b = g.add_vertex(zx.VertexType.Z, phase=Fraction(1, 4))
            x = g.add_vertex(zx.VertexType.X, phase=Fraction(1, 4))
            for u, v, t in [(bi,a,1),(a,b,1),(a,x,2),(b,x,2),(x,bo,1)]:
                g.add_edge((u,v), t)
            g.set_inputs((bi,)); g.set_outputs((bo,))
            for method in ['spider_fusion', 'full_reduce']:
                load(db, g)
                getattr(db, method)('audit')
                results[method + '_zx_parallel_h_semantic'] = _quimb_equal(g, export(db))
                db.empty_graphdb('audit')

            # The supplementary rule assumes H-connected graph-like input,
            # but has no edge-type guard on this valid ZX diagram.
            g = zx.Graph()
            bi = g.add_vertex(zx.VertexType.BOUNDARY)
            bo = g.add_vertex(zx.VertexType.BOUNDARY)
            c = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
            d = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
            a = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
            b = g.add_vertex(zx.VertexType.Z, phase=Fraction(3,4))
            for u,v in [(bi,c),(d,bo),(a,c),(b,c),(a,d),(b,d)]: g.add_edge((u,v))
            g.set_inputs((bi,)); g.set_outputs((bo,))
            load(db,g)
            results['supplementarity_simple_applications'] = db.supplementarity_simp('audit')
            results['supplementarity_simple_semantic'] = _quimb_equal(g,export(db))
            db.empty_graphdb('audit')

            # Raw multigraph round-trip should preserve both typed wires.
            g = zx.Graph()
            a = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
            b = g.add_vertex(zx.VertexType.Z, phase=Fraction(1,4))
            g.add_edge((a,b),1); g.add_edge((a,b),2)
            load(db,g)
            results['parallel_export_edges'] = export(db).num_edges()
            db.empty_graphdb('audit')

            try:
                db.turn_hadamard_gates_into_edges('audit')
            except Exception as exc:
                results['hadamard_conversion_error'] = repr(exc)
        finally:
            for gid in ['audit', 'other']:
                db.empty_graphdb(gid)
    result_dir = Path('evaluation/audit_results')
    result_dir.mkdir(exist_ok=True)
    (result_dir / 'system_probes.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
