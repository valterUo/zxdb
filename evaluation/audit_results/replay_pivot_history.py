import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['ZXDB_URI'] = 'bolt://localhost:7688'
plan = json.loads(Path('results-study/plan.json').read_text())
out = Path('evaluation/audit_results/pivot-history-repro').resolve()
(out / 'blocks').mkdir(parents=True, exist_ok=True)
(out / 'inputs').mkdir(exist_ok=True)
(out / 'plan.json').write_text(json.dumps(plan))
history = json.loads(Path('results-study/campaign.json').read_text())['outcomes'][:11]
for entry in history:
    index = next(i for i, c in enumerate(plan['cases']) if c['id'] == entry['id'])
    print(index, entry['id'], flush=True)
    with (out / 'blocks' / (entry['id'] + '.log')).open('w') as log:
        proc = subprocess.run([sys.executable, '-m', 'evaluation.research', '_worker',
                               '--plan', str(out / 'plan.json'), '--index', str(index),
                               '--output', str(out)], stdout=log, stderr=subprocess.STDOUT, timeout=90)
    if proc.returncode:
        raise RuntimeError('Failed ' + entry['id'])
