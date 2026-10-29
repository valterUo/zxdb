"""Apply the user's five-minute engine-run skip policy to the running campaign."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

import psutil

ROOT = Path(__file__).resolve().parent
LIMIT = 300


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def active():
    state = json.loads((ROOT/'progress.json').read_text())
    current = state.get('current')
    if state['status'] != 'running' or not current:
        return state, None
    if any(c['index'] == current['index'] for c in state['completed']):
        return state, None
    case = ROOT/current['directory']
    log = case/'run.log'
    if not log.exists():
        return state, None
    content = log.read_text(errors='replace')
    events = list(re.finditer(r'^ENGINE (\w+) repetition=(\d+) (.+)$', content, re.MULTILINE))
    if not events or events[-1].group(3) != 'starting':
        return state, None
    event = events[-1]
    return state, dict(case=case, engine=event.group(1), repetition=int(event.group(2)),
                       key=(current['index'], event.start()), started=log.stat().st_mtime)


def targets(state, case):
    supervisor = psutil.Process(state['pid'])
    if 'continue_single_rules.py' not in ' '.join(supervisor.cmdline()):
        raise RuntimeError('Supervisor identity changed')
    matches = []
    for process in supervisor.children(recursive=True):
        try:
            command = process.cmdline()
            if 'evaluation.rerun_three_engines' in command and str(case) in command:
                matches.append(process)
        except psutil.NoSuchProcess:
            pass
    # On Windows the venv launcher owns the actual Python process. Stop only
    # the leaf worker; its launcher then exits and wakes the supervisor.
    parents = {process.ppid() for process in matches}
    leaves = [process for process in matches if process.pid not in parents]
    if len(leaves) != 1:
        raise RuntimeError(f'Expected one matching case worker, found {len(leaves)}')
    return leaves[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Read-only diagnosis; never stop a process')
    args = parser.parse_args()
    tracked_key, started = None, None
    if not args.inspect:
        write(ROOT/'engine-run-policy.json', dict(created=datetime.now(timezone.utc).isoformat(),
              seconds=LIMIT, pid=os.getpid(), scope='Every engine run, warmup and measured repetitions, in current and remaining cases',
              action='Interrupt the entire case; preserve partial observations; continue subsequent cases',
              timing='Observed engine-run wall time, including any verification before the completion log event',
              measured_repetition_limit_seconds=LIMIT))
    while True:
        try:
            state, event = active()
            if state['status'] != 'running':
                print('Campaign no longer running; engine-run monitor exiting.', flush=True)
                return
            if event:
                if event['key'] != tracked_key:
                    tracked_key, started = event['key'], event['started']
                elapsed = time.time()-started
                if args.inspect:
                    process = targets(state, event['case'])
                    print(json.dumps(dict(case=state['current']['index'], engine=event['engine'],
                                          repetition=event['repetition'], run_seconds=elapsed, limit=LIMIT, worker_pid=process.pid)))
                    return
                if elapsed >= LIMIT:
                    process = targets(state, event['case'])
                    latest, fresh = active()
                    if fresh and fresh['key'] == tracked_key and latest['pid'] == state['pid']:
                        decision = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            case=state['current'], reason='User-requested engine-run wall-time limit exceeded',
                            engine=event['engine'], repetition=event['repetition'], elapsed_seconds=elapsed, limit_seconds=LIMIT,
                            worker_pid=process.pid, command=process.cmdline())
                        write(event['case']/'engine-run-skip.json', decision)
                        process.kill()
                        print(json.dumps(decision), flush=True)
                        # Allow the original supervisor to record, recover and advance.
                        time.sleep(10)
            elif args.inspect:
                print('No active engine run.')
                return
            if not args.inspect:
                write(ROOT/'engine-run-monitor-status.json', dict(pid=os.getpid(),
                      updated=datetime.now(timezone.utc).isoformat(),
                      current_index=state.get('current', {}).get('index') if state.get('current') else None,
                      monitoring_engine_run=event is not None, repetition=event['repetition'] if event else None))
        except (OSError, ValueError, RuntimeError, psutil.Error) as exc:
            print(f'{datetime.now(timezone.utc).isoformat()} {type(exc).__name__}: {exc}', flush=True)
            if args.inspect:
                raise
        time.sleep(5)


if __name__ == '__main__':
    main()
