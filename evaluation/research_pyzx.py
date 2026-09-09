"""Resource-isolated PyZX attempt. Process startup and IPC are not reduction time."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time


def isolated_reference(graph, rule, timeout_s, max_rss_mb):
    if timeout_s <= 0 or max_rss_mb <= 0:
        raise ValueError('PyZX resource limits must be positive')
    with tempfile.TemporaryDirectory(prefix='zxdb-pyzx-') as folder:
        root = Path(folder)
        with (root / 'input.pkl').open('wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        started = time.monotonic()
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        try:
            process = subprocess.run([sys.executable, '-m', 'evaluation.research_pyzx', folder, rule,
                                      str(timeout_s), str(max_rss_mb)], env=env,
                                     timeout=timeout_s + 10, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return None, dict(status='timeout', wall_s=time.monotonic()-started,
                              reason='Isolated process wall timeout', reduce_s=None)
        if not (root / 'result.json').exists():
            raise RuntimeError(f'PyZX child failed ({process.returncode}): {process.stderr[-2000:]}')
        info = json.loads((root / 'result.json').read_text())
        info['wall_s'] = time.monotonic()-started
        info['execution_mode'] = 'isolated_process'
        result = None
        if info['status'] == 'ok':
            with (root / 'output.pkl').open('rb') as f:
                result = pickle.load(f)
        return result, info


def child(folder, rule, timeout_s, max_rss_mb):
    import threading
    import psutil
    root = Path(folder)
    process = psutil.Process()
    parent = process.parent()
    started = time.monotonic()
    peak = [0]
    done = threading.Event()

    def save(value):
        temporary = root / 'result.tmp'
        temporary.write_text(json.dumps(value))
        temporary.replace(root / 'result.json')

    def watch():
        while not done.wait(.05):
            if parent is not None and not parent.is_running():
                os._exit(0)
            rss = process.memory_info().rss
            peak[0] = max(peak[0], rss)
            status = ('memory_limit' if rss > max_rss_mb * 1024**2 else
                      'timeout' if time.monotonic()-started > timeout_s else None)
            if status:
                save(dict(status=status, reduce_s=None, peak_rss_bytes=peak[0],
                          reason='PyZX process resource budget exceeded'))
                os._exit(0)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        from evaluation.research_core import reference
        with (root / 'input.pkl').open('rb') as f:
            original = pickle.load(f)
        t = time.perf_counter(); graph = original.copy(); setup_s = time.perf_counter()-t
        t = time.perf_counter(); reference(graph, rule, float('inf')); reduce_s = time.perf_counter()-t
        with (root / 'output.pkl').open('wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        done.set(); thread.join()
        save(dict(status='ok', reduce_s=reduce_s, setup_s=setup_s,
                  peak_rss_bytes=max(peak[0], process.memory_info().rss)))
    except MemoryError:
        done.set(); thread.join()
        save(dict(status='memory_limit', reduce_s=None, reason='Python MemoryError'))


if __name__ == '__main__':
    child(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
