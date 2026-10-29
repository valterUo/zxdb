"""Read-only live terminal view. Run this script; Ctrl+C closes only the viewer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else f'{minutes}m {seconds:02d}s'


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may be in the middle of its final line.
    return rows


def render():
    plan = json.loads((ROOT/'plan.json').read_text(encoding='utf-8'))
    state = json.loads((ROOT/'progress.json').read_text(encoding='utf-8'))
    now = time.time()
    done = state['completed']
    for case in done:
        if case.get('directory') and any((ROOT/case['directory']/name).exists()
                for name in ('warmup-skip.json', 'engine-run-skip.json')):
            case['status'] = 'skipped'
    total = len(plan['jobs'])
    succeeded = sum(c['status'] == 'complete' for c in done)
    skipped = sum(c['status'] == 'skipped' for c in done)
    failed = len(done)-succeeded-skipped
    percent = len(done)/total
    bar = '#' * int(percent*32) + '-' * (32-int(percent*32))
    elapsed = now-datetime.fromisoformat(plan['created']).timestamp()
    lines = ['ZXdb benchmark progress', '='*76,
             f"Campaign: {state['status'].upper()}   Stage: {state['stage']}",
             f'[{bar}] {len(done)}/{total} cases resolved ({percent:.1%})',
             f'Successful: {succeeded}   Failed/interrupted: {failed}   Skipped: {skipped}   Remaining: {total-len(done)}',
             f'Campaign elapsed: {duration(elapsed)}', '',
             'Suite                   Successful   Failed   Skipped   Planned']
    for label, is_full in [('Full reduce', True), ('Single rules', False)]:
        planned = sum((j['rule']=='full_reduce') == is_full for j in plan['jobs'])
        selected = [c for c in done if (c['rule']=='full_reduce') == is_full]
        ok = sum(c['status']=='complete' for c in selected)
        skip = sum(c['status']=='skipped' for c in selected)
        lines.append(f'{label:24}{ok:>10}{len(selected)-ok-skip:>9}{skip:>10}{planned:>10}')
    lines += ['', 'CURRENT WORKLOAD']
    current = state.get('current')
    if current:
        spec = current['spec']
        case = ROOT/current['directory']
        lines.append(f"Case {current['index']+1}/{total}: {spec['family']} | size {spec['size']} | seed {spec['seed']}")
        manifest = case/'manifest.json'
        if manifest.exists():
            lines.append(f'Case elapsed: {duration(now-manifest.stat().st_mtime)}')
        logfile = case/'run.log'
        if logfile.exists():
            log = logfile.read_text(encoding='utf-8', errors='replace').splitlines()
            events = [line for line in log if line.startswith('ENGINE ')]
            if events:
                lines.append('Latest engine event: '+events[-1])
                lines.append(f'Last log activity: {duration(now-logfile.stat().st_mtime)} ago')
            rows = read_rows(case/'engine-observations.jsonl')
            lines.append(f'Engine trials recorded for this case: {len(rows)}/12')
            lines.append('Engine       Warmup    Measured repetitions recorded')
            for engine in ('pyzx', 'cypher', 'hybrid'):
                selected = [r for r in rows if r['engine']==engine]
                warmup = next((r['status'] for r in selected if r['warmup']), 'pending')
                measured = ', '.join(f"{r['repetition']}:{r['status']}" for r in selected if not r['warmup']) or 'pending'
                lines.append(f'{engine:12} {warmup:12} {measured}')
        if any(c['index']==current['index'] for c in done):
            lines.append('Case finished; supervisor is reporting or recovering before the next case.')
    else:
        lines.append('No timing case active.')
    lines += ['', 'RECENT CASES']
    for case in done[-4:]:
        spec = case['spec']
        lines.append(f"{case['index']+1:3}  {case['status']:8} {spec['family']:23} size={spec['size']:<6} seed={spec['seed']}  {duration(case['wall_seconds'])}")
    if state.get('error'):
        lines += ['', 'SUPERVISOR ERROR: '+state['error']]
    fixtures = [r for r in read_rows(ROOT/'correctness/observations.jsonl') if r['stage']=='correctness']
    counts = Counter(r['status'] for r in fixtures)
    if counts:
        lines += ['', 'Correctness: '+', '.join(f'{count} {status}' for status, count in sorted(counts.items()))]
    lines += ['', 'Case percentage is not time percentage: large workloads take much longer.',
              'inconclusive = semantic check unresolved; error/failed = unsuccessful trial.',
              'Engine-run cutoff: 5 minutes per engine, for warmups and measured runs.',
              'Read-only view; no database queries. Refreshes every 5 seconds.',
              'Ctrl+C closes this viewer and leaves the benchmark running.',
              'Updated: '+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='Print a single snapshot')
    args = parser.parse_args()
    if not args.once and sys.stdout.isatty():
        os.system('')  # Enable ANSI sequences in Windows terminals.
    try:
        while True:
            try:
                screen = render()
            except (OSError, ValueError) as exc:
                screen = f'Waiting for readable progress: {exc}'
            if not args.once and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(screen, flush=True)
            if args.once:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        print('\nViewer closed; benchmark continues.')


if __name__ == '__main__':
    main()
