"""Audit existing rule fixtures against their INPUT, independently of the verdict.

Run only against a disposable, otherwise empty Memgraph instance: the existing
run_case importer clears the database. This is a diagnostic, not a test verdict.
"""
import json
from pathlib import Path

from evaluation.cases import RULES
from evaluation.harness import GRAPH_ID, run_case, _quimb_equal
from zxdb.zxdb import ZXdb


def main():
    results = []
    with ZXdb() as db:
        if db.node_count():
            raise RuntimeError('Audit requires an empty disposable database')
        for rule, spec in RULES.items():
            for case in spec['cases']:
                name, builder = case[:2]
                original = builder()
                result = run_case(db, rule, name, original.copy(),
                                  getattr(db, spec['db_method']), spec['pyzx'],
                                  **(case[2] if len(case) > 2 else {}))
                if result['error'] is None:
                    reduced = db.export_graphdb_to_zx_graph(GRAPH_ID, 'example.json')
                    result['audit_vs_original'] = _quimb_equal(reduced, original)
                results.append(result)
                print(rule, name, result.get('audit_vs_original'), flush=True)
        db.empty_graphdb(GRAPH_ID)
    result_dir = Path('evaluation/audit_results')
    result_dir.mkdir(exist_ok=True)
    (result_dir / 'original_semantics.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
