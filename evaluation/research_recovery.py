"""Durable checkpoints and explicit Docker recovery for dedicated campaigns."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse
from contextlib import contextmanager


@contextmanager
def campaign_lock(directory):
    """OS-released lock: a terminated parent cannot leave a stale active lock."""
    path = Path(directory).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another runner is using this output directory') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def records(path, repair=False):
    """Only an incomplete final JSONL record may be discarded after interruption."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    result, offset = [], 0
    for line in data.splitlines(keepends=True):
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if offset + len(line) != len(data) or line.endswith(b'\n'):
                raise ValueError(f'Corrupt completed record in {path}')
            if repair:
                backup = path.with_name(path.name + f'.partial-{time.time_ns()}')
                backup.write_bytes(line)
                with path.open('r+b') as f:
                    f.truncate(offset)
            break
        offset += len(line)
    if repair and result and offset == len(data) and not data.endswith(b'\n'):
        with path.open('ab') as f:
            f.write(b'\n')
    return result


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def ready(uri, auth):
    from neo4j import GraphDatabase, Query
    try:
        with GraphDatabase.driver(uri, auth=auth, connection_timeout=2,
                                  connection_acquisition_timeout=3, max_transaction_retry_time=0) as driver:
            with driver.session() as s:
                return s.run(Query('RETURN 1 AS n', timeout=2)).single()['n'] == 1
    except Exception:
        return False


def check_container(container, uri):
    parsed = urlparse(uri)
    if parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Docker restart requires a local ZXDB_URI')
    ports = json.loads(subprocess.check_output(
        ['docker', 'inspect', '--format', '{{json .HostConfig.PortBindings}}', container], text=True, timeout=15))
    if not any(str(p['HostPort']) == str(parsed.port or 7687) for p in (ports or {}).get('7687/tcp', []) or []):
        raise ValueError('Selected container does not publish the ZXDB_URI port')


def recover(container, uri, auth, directory, reason, timeout=90):
    """Restart the explicitly selected container, wait for Bolt, clean owned graphs."""
    check_container(container, uri)
    path = Path(directory) / 'recovery.jsonl'
    append(path, dict(event='restart_requested', reason=reason, container=container, time=time.time()))
    try:
        result = subprocess.run(['docker', 'restart', container], capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        deadline = time.monotonic() + timeout
        while not ready(uri, auth):
            if time.monotonic() >= deadline:
                raise TimeoutError('Memgraph did not become ready after restart')
            time.sleep(1)
        cleanup_owned(uri, auth, directory)
        append(path, dict(event='restart_ready', time=time.time()))
    except Exception as exc:
        append(path, dict(event='restart_failed', error=str(exc), time=time.time()))
        raise


def cleanup_owned(uri, auth, directory):
    """Delete only exact graph IDs recorded by this campaign's workers."""
    from neo4j import GraphDatabase, Query
    owned = {r['graph_id'] for p in (Path(directory) / 'blocks').glob('*.jsonl')
             for r in records(p) if r.get('kind') == 'environment' and 'graph_id' in r}
    with GraphDatabase.driver(uri, auth=auth, connection_timeout=3) as driver:
        with driver.session() as s:
            foreign = s.run(Query('MATCH(n) WHERE n.graph_id IS NULL OR NOT n.graph_id IN $ids '
                                  'RETURN count(n) AS n', timeout=10), ids=list(owned)).single()['n']
            if foreign:
                raise RuntimeError('Database contains unrelated data; recovery refuses to delete it')
            for gid in sorted(owned):
                s.run(Query('MATCH(n:Node {graph_id:$gid}) DETACH DELETE n', timeout=10), gid=gid).consume()
