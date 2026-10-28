"""#608 cross-host supervised cell: labnode3 A2 coordinator + desktop 3090 worker.

Validates the polled-heartbeat fix live: SIGKILLing the remote rank-1
worker must surface as a catchable peer-dead classification within the
heartbeat deadline (coordinator process ALIVE — the #637 finding was that
the NCCL watchdog killed it instead), /health honest, bounded operator
rejoin reaching fresh generations with a respawned remote worker, and the
generation bound (3) going terminal with no fourth respawn.

Failure injection kills the remote worker by /proc-verified PID matched on
its TORCHELASTIC_RUN_ID environ — never an unscoped pattern kill.
"""
import json
import os
import signal
import subprocess
import time
from http.client import HTTPConnection
from pathlib import Path

PYTHON = '/home/damen/dllm-dev/venv/bin/python'
WORKER_HOST = 'rocky10-desktop'
MODEL = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/'
         'snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987')
WORKER_MODEL = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/'
                'snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987')
A2 = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
RTX3090 = 'GPU-24ef3d4c-51dc-d004-4150-d6a70b05e069'
PORT = 8410
OUT = '/tmp/g608x/cross-host-cell.json'
LOG = '/tmp/g608x/serve-supervised.log'

events = []


def emit(**event):
    print(json.dumps(event), flush=True)


def record(event):
    events.append(event)
    emit(**event)
    # Evidence persistence is incremental: any later crash (ours or the
    # coordinator's) still leaves everything observed so far on disk.
    Path(OUT).write_text(json.dumps(events, indent=1))


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


def wait_health(target, deadline_seconds=420):
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


REMOTE_PID_SCRIPT = (
    "for pid in $(pgrep -f damendaemon.distributed_worker); do"
    " if tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null"
    " | grep -q '^TORCHELASTIC_RUN_ID={prefix}'; then echo $pid; fi;"
    " done")


def remote_worker_pids(run_id_prefix):
    script = REMOTE_PID_SCRIPT.format(prefix=run_id_prefix)
    result = subprocess.run(['ssh', WORKER_HOST, script], capture_output=True,
                           text=True, timeout=15, check=False)
    return [int(pid) for pid in result.stdout.split() if pid.isdigit()]


def kill_remote_worker(reason, run_id_prefix):
    pids = remote_worker_pids(run_id_prefix)
    if not pids:
        return dict(reason=reason, killed=[], note='no matching remote worker')
    for pid in pids:
        subprocess.run(['ssh', WORKER_HOST, f'kill -9 {pid}'],
                       capture_output=True, timeout=10, check=False)
    time.sleep(1)
    return dict(reason=reason, killed=pids)


def completion(prompt='The capital of France is'):
    status, body = request('POST', '/v1/completions',
                           {'model': 'model', 'prompt': prompt,
                            'max_tokens': 12, 'temperature': 0})
    text = ''
    if status == 200 and body.get('choices'):
        text = body['choices'][0].get('text', '')
    return dict(status=status, text=text,
                completion_tokens=body.get('usage', {}).get('completion_tokens'))


def alive(launcher):
    return launcher.poll() is None


def main():
    launcher = subprocess.Popen(
        [PYTHON, '-m', 'damendaemon', 'distributed-serve', MODEL,
         '--supervise', '--port', str(PORT), '--context', '512',
         '--batch-size', '1', '--quantization', 'none',
         '--coordinator-gpu', A2,
         '--worker-host', WORKER_HOST,
         '--worker-gpu', RTX3090,
         '--worker-python', '/home/damen/3090-damendaemon/venv/bin/python',
         '--worker-repo', '/tmp/608fix',
         '--worker-model-path', WORKER_MODEL,
         '--rendezvous-host', '10.30.100.3',
         '--coordinator-ifname', 'enp6s18',
         '--worker-ifname', 'enp72s0',
         '--heartbeat-deadline', '10'],
        env=dict(os.environ, PYTHONPATH='/tmp/608fix'),
        stdout=open(LOG, 'w'), stderr=subprocess.STDOUT)
    run_id_prefix = f'supervised-{launcher.pid}-'
    try:
        healthy = wait_health('healthy')
        record(dict(event='startup', health=healthy,
                           coordinator_pid=launcher.pid,
                           note='generation 1 cross-host pair up (A2 '
                                'coordinator, 3090 worker over SSH); the '
                                'all_gather transport-version agreement '
                                'passed at pair construction'))
        first = completion()
        record(dict(event='completion', generation=1, **first))
        gen1_pids = remote_worker_pids(run_id_prefix)
        record(dict(event='remote_worker_pids', generation=1, pids=gen1_pids))
        # Failure injection: SIGKILL the real remote worker.
        kill_started = time.monotonic()
        record(dict(event='kill', generation=1,
                           **kill_remote_worker('remote rank-1 SIGKILL', run_id_prefix)))
        failed = wait_health('failed', 120)
        classify_elapsed = round(time.monotonic() - kill_started, 1)
        record(dict(event='health_failed', observed=failed,
                           seconds_from_kill_to_failed_state=classify_elapsed,
                           coordinator_alive=alive(launcher),
                           note='THE fix claim: coordinator survived; '
                                'peer-dead classified via the polled '
                                'heartbeat inside its 10s deadline '
                                '(heartbeat interval 5s), not the '
                                'process-fatal 30s NCCL group timeout'))
        status, body = request('GET', '/health')
        record(dict(event='failed_health_detail', status=status, body=body))
        # Bounded operator rejoin: fresh remote worker + pair, empty slots.
        status, body = request('POST', '/v1/distributed/rejoin')
        record(dict(event='rejoin', generation=2, status=status, body=body))
        healthy2 = wait_health('healthy')
        record(dict(event='health_after_rejoin', observed=healthy2))
        second = completion('Water boils at a temperature of')
        record(dict(event='completion', generation=2, **second))
        gen2_pids = remote_worker_pids(run_id_prefix)
        record(dict(event='remote_worker_pids', generation=2, pids=gen2_pids,
                           note='fresh remote worker process, distinct from '
                                f'generation 1: {gen2_pids != gen1_pids}'))
        # Second failure + rejoin: generation 3.
        record(dict(event='kill', generation=2,
                           **kill_remote_worker('second remote SIGKILL', run_id_prefix)))
        failed2 = wait_health('failed', 120)
        record(dict(event='health_failed', observed=failed2,
                           coordinator_alive=alive(launcher)))
        status, body = request('POST', '/v1/distributed/rejoin')
        record(dict(event='rejoin', generation=3, status=status, body=body))
        healthy3 = wait_health('healthy')
        record(dict(event='health_after_rejoin', observed=healthy3))
        third = completion('The three primary colors are')
        record(dict(event='completion', generation=3, **third))
        # Third failure: the generation bound (default 3) must go terminal.
        record(dict(event='kill', generation=3,
                           **kill_remote_worker('third remote SIGKILL', run_id_prefix)))
        failed3 = wait_health('failed', 120)
        record(dict(event='health_failed', observed=failed3,
                           coordinator_alive=alive(launcher)))
        status, body = request('POST', '/v1/distributed/rejoin')
        record(dict(event='rejoin_refused_past_bound', status=status, body=body,
                           note='409 expected: generation bound 3 terminal; '
                                'no automatic restart, no request replay'))
        time.sleep(5)
        record(dict(event='post_bound_remote_worker_check',
                           remote_pids_still_running=remote_worker_pids(run_id_prefix)))
        record(dict(event='final_coordinator_alive', alive=alive(launcher)))
    finally:
        if alive(launcher):
            launcher.terminate()
            try:
                launcher.wait(60)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(10)
        for pid in remote_worker_pids(run_id_prefix):
            subprocess.run(['ssh', WORKER_HOST, f'kill -9 {pid}'],
                           capture_output=True, timeout=10, check=False)
    Path(OUT).write_text(json.dumps(events, indent=1))
    print(json.dumps({'report': OUT, 'events': len(events)}), flush=True)


if __name__ == '__main__':
    main()
