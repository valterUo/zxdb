"""Append missing focused families while preserving existing checkpoint identities."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation.research_adaptive import ordered_cases
from evaluation.research_recovery import campaign_lock
from evaluation.research_workloads import make_plan

# Filled by the audited migration: these exact changes affect orchestration only.
AUDITED_SOURCE_UPDATES = {'evaluation/research.py': [['71d6367d0dcee73a28dc355ee70fdd3f532f62f82516a772c9e8a2ee63672c87', '953c83e2393b099f96a46325439c9fe69a985c00e4cea4d28788774812b8dab9']], 'evaluation/research_adaptive.py': [['ff1b5756aef737403b104589726987c4c6606f23b9d6873c0f6a7fe4f13871c4', 'bc83cb146214b826b198e68e5202d5e78ce8dce6514f9d06d3fe7f09c284edba']]}
REPORT_ONLY = {'evaluation/research_report.py', 'evaluation/research_scale.py',
               'evaluation/research_extend.py'}


def compatible_sources(saved, current):
    for name in set(saved) | set(current):
        if name in REPORT_ONLY or saved.get(name) == current.get(name):
            continue
        if [saved.get(name), current.get(name)] not in AUDITED_SOURCE_UPDATES.get(name, []):
            return False
    return True


def extended_plan(saved, candidates):
    if saved.get('profile') != 'focused':
        raise ValueError('Only focused campaigns can use this extension')
    existing_families = {s['family'] for s in saved['cases']}
    additions = [s for s in candidates['cases'] if s['family'] not in existing_families]
    if not additions:
        return saved
    # Keep all existing definitions, IDs and worker indices exactly unchanged.
    result = dict(saved, cases=saved['cases'] + additions)
    tail = dict(saved, cases=additions)
    tail.pop('case_order', None)
    result['case_order'] = ordered_cases(saved) + [len(saved['cases'])+i for i in ordered_cases(tail)]
    return result


def extend(plan_path, output):
    directory = Path(output)
    plan_path = Path(plan_path)
    with campaign_lock(directory):
        stored_path = directory/'plan.json'
        saved = json.loads(stored_path.read_text())
        if json.loads(plan_path.read_text()) != saved:
            raise ValueError('External plan must match the saved campaign before extension')
        candidate = make_plan('focused')
        # Existing families must still have the exact generator specifications.
        candidate_cases = {s['id']: s for s in candidate['cases']}
        if any(candidate_cases.get(s['id']) != s for s in saved['cases']):
            raise ValueError('Focused generator differs from saved cases; cannot extend')
        updated = extended_plan(saved, candidate)
        if updated == saved:
            print('All focused families are already present')
            return
        archive = directory/'plan-history'; archive.mkdir(exist_ok=True)
        old_bytes = stored_path.read_bytes()
        digest = hashlib.sha256(old_bytes).hexdigest()[:16]
        backup = archive/(digest+'.json')
        if not backup.exists():
            backup.write_bytes(old_bytes)
        text = json.dumps(updated, indent=2, sort_keys=True)
        temporary = directory/'plan.extending.json'
        temporary.write_text(text, encoding='utf-8'); temporary.replace(stored_path)
        if plan_path.resolve() != stored_path.resolve():
            plan_path.write_text(text, encoding='utf-8')
        print(f'Extended {len(saved["cases"])} to {len(updated["cases"])} cases; existing blocks untouched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extend(args.plan, args.output)
