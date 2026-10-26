"""#608 live supervised two-rank cell on the labnode3 A30+A2 pair.

Coordinator process + supervisor-spawned rank-1 worker child (one per
generation). The cell exercises the paths that stayed CPU-bus-verified
only: real dead-peer detection after a killed worker, honest failed
/health, the real respawn path on operator rejoin, resume-with-empty-slots
serving after rejoin, the generation bound (default 3) going terminal, and
the live all_gather transport-version agreement at pair startup.

Failure injection is by explicit worker-child PID (pgrep'd from the exact
module cmdline, then verified against /proc cmdline) — never a pattern
kill that could hit this driver or the coordinator.
"""
import json
import os
import signal
import subprocess
import time
from http.client import HTTPConnection
from pathlib import Path

PYTHON = '/home/damen/dllm-dev/venv/bin/python'
MODEL = str(next(Path('/home/damen/.cache/huggingface/hub/'
                      'models--Qwen--Qwen2.5-0.5B/snapshots').glob('*')))
PORT = 8400
OUT = '/tmp/g608/supervised-cell.json'
LOG = '/tmp/g608/serve-supervised.log'


def emit(**event):
    print(json.dumps(event), flush=True)


def request(method, path, payload=None):
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {'Content-Type': 'application/json',
                                        'Content-Length': str(len(body))}
    conn = HTTPConnection('127.0.0.1', PORT, timeout=110)
    conn.request(method, path, body, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        parsed = {'raw': data.decode('utf-8', 'replace')[:500]}
    return response.status, parsed


def wait_health(target, deadline_seconds=240):
    """Poll /health until its status/healthy flag matches the target."""
    deadline = time.monotonic() + deadline_seconds
    last = None
    while time.monotonic() < deadline:
        try:
            status, body = request('GET', '/health')
            last = (status, body)
            if target == 'healthy' and status == 200 and body.get('healthy', True):
                return last
            if target == 'failed' and (status != 200 or body.get('healthy') is False):
                return last
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit(f'/health never reached {target}: {last}')


def worker_pid():
    out = subprocess.run(['pgrep', '-f', 'damendaemon.distributed_worker'],
                         capture_output=True, text=True, timeout=10)
    for pid in out.stdout.split():
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8', 'replace')
        except OSError:
            continue
        if 'damendaemon.distributed_worker' in cmdline and 'pgrep' not in cmdline:
            return int(pid)
    return None


def completion(prompt='The capital of France is'):
    status, body = request('POST', '/v1/completions',
                           {'model': 'model', 'prompt': prompt,
                            'max_tokens': 12, 'temperature': 0})
    text = ''
    if status == 200 and body.get('choices'):
        text = body['choices'][0].get('text', '')
    return dict(status=status, text=text,
                completion_tokens=body.get('usage', {}).get('completion_tokens'))


def kill_worker(reason):
    pid = worker_pid()
    if pid is None:
        return dict(reason=reason, killed=None, note='no worker child found')
    os.kill(pid, signal.SIGKILL)
    time.sleep(1)
    return dict(reason=reason, killed=pid)


def main():
    events = []
    launcher = subprocess.Popen(
        [PYTHON, '-m', 'damendaemon', 'distributed-serve', MODEL,
         '--supervise', '--port', str(PORT), '--context', '512',
         '--batch-size', '1', '--quantization', 'none'],
        env=dict(os.environ), stdout=open(LOG, 'w'), stderr=subprocess.STDOUT)
    try:
        healthy = wait_health('healthy', 420)
        events.append(dict(event='startup', health=healthy,
                           note='generation 1 pair up; the all_gather '
                                'transport-version agreement passed at pair '
                                'construction (same venv both ranks)'))
        first = completion()
        events.append(dict(event='completion', generation=1, **first))
        # Failure injection: kill the real worker child.
        events.append(dict(event='kill', generation=1, **kill_worker('rank-1 worker SIGKILL')))
        failed = wait_health('failed', 240)
        events.append(dict(event='health_failed', observed=failed,
                           note='HTTP stays up in the failed state; /health is honest'))
        # Bounded operator rejoin: fresh worker + pair, empty slots.
        status, body = request('POST', '/v1/distributed/rejoin')
        events.append(dict(event='rejoin', generation=2, status=status, body=body))
        healthy2 = wait_health('healthy', 420)
        events.append(dict(event='health_after_rejoin', observed=healthy2))
        second = completion('Water boils at a temperature of')
        events.append(dict(event='completion', generation=2, **second))
        # Second failure + rejoin: generation 3.
        events.append(dict(event='kill', generation=2, **kill_worker('second rank-1 SIGKILL')))
        failed2 = wait_health('failed', 240)
        events.append(dict(event='health_failed', observed=failed2))
        status, body = request('POST', '/v1/distributed/rejoin')
        events.append(dict(event='rejoin', generation=3, status=status, body=body))
        healthy3 = wait_health('healthy', 420)
        events.append(dict(event='health_after_rejoin', observed=healthy3))
        third = completion('The three primary colors are')
        events.append(dict(event='completion', generation=3, **third))
        # Third failure: the generation bound (default 3) must go terminal —
        # a further rejoin is refused with 409, no fourth respawn.
        events.append(dict(event='kill', generation=3, **kill_worker('third rank-1 SIGKILL')))
        failed3 = wait_health('failed', 240)
        events.append(dict(event='health_failed', observed=failed3))
        status, body = request('POST', '/v1/distributed/rejoin')
        events.append(dict(event='rejoin_refused_past_bound', status=status, body=body,
                           note='409 expected: generation bound 3 is terminal; '
                                'no automatic restart, no request replay'))
        time.sleep(5)
        events.append(dict(event='post_bound_worker_check',
                           worker_pid_running=worker_pid() is not None))
    finally:
        launcher.terminate()
        try:
            launcher.wait(60)
        except subprocess.TimeoutExpired:
            launcher.kill()
            launcher.wait(10)
        pid = worker_pid()
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
    Path(OUT).write_text(json.dumps(events, indent=1))
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
