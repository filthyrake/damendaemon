"""Two-rank A→B→A resident model switching gate (#489).

Starts the guarded two-rank distributed-serve with two resident model aliases
and drives one greedy generation per alias in A→B→A order. The B and second-A
requests exercise the live SWITCH stage command on both ranks (drain,
peer-verified identity table, epoch-advancing commit). Passes when all three
generations return finite text, the two A generations are byte-identical, and
/v1/models/status records the switches with the expected final residency.

By default both aliases point at the same real checkpoint: the identity
contract requires identical decoder configs, and the available second
checkpoint (Qwen2.5-0.5B base) differs in eos_token_id semantics, so
distinct-weights switching is correctly rejected by that gate. --second can
point at any compatible distinct checkpoint when one is staged. Requires the
two-device guard: CUDA_VISIBLE_DEVICES is a comma pair, one rank per device.
Timing is recorded per request; shared-host latencies are not performance
evidence.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.split_gguf import save

SNAPSHOT = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/'
            'snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
SECOND = SNAPSHOT
PORT = 8125
MASTER_PORT = '29531'
PROMPTS = {'model': 'What is 17 * 23? Answer with one number.',
           'alt': 'What is 19 * 21? Answer with one number.'}


def health(deadline):
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return True
        except Exception:
            time.sleep(2)
    return False


def generate(alias, max_tokens=16):
    body = json.dumps(dict(model=alias, prompt=PROMPTS[alias], max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:200]}') from None
    return dict(alias=alias, text=payload['choices'][0]['text'],
                tokens=payload['usage']['completion_tokens'],
                seconds=time.monotonic() - started)


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=5) as response:
            payload = json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--second', default=SECOND,
                        help='path of the second resident model (alias alt)')
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 2 or any(not uuid.startswith('GPU-') for uuid in visible):
        raise ValueError('the switch gate requires the two-device guard')
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report = dict(completed=False, model=SNAPSHOT, second=args.second, devices=visible,
                  torch=torch.__version__, quantization='q4_0', context=512, generations=[],
                  statuses=[])
    save(args.output, report)
    server = None
    environment = dict(os.environ)
    try:
        environment['PYTHONUNBUFFERED'] = '1'
        server = subprocess.Popen(
            [sys.executable, '-u', '-m', 'torch.distributed.run', '--nproc-per-node', '2',
             '--master-port', MASTER_PORT,
             '-m', 'damendaemon', 'distributed-serve', SNAPSHOT,
             '--model', f'alt={args.second}',
             '--quantization', 'q4_0', '--context', '512', '--port', str(PORT),
             '--batch-size', '4', '--queue-timeout', '300', '--idle-timeout', '600'],
            stdout=open(args.output.with_suffix('.serve.log'), 'w'), stderr=subprocess.STDOUT)
        if not health(time.monotonic() + 900):
            raise RuntimeError('two-rank service did not become healthy within 900 seconds')
        first = generate('model')          # resident on A
        report['statuses'].append(dict(phase='A1', status=status()))
        second = generate('alt')           # SWITCH A→B on both ranks
        report['statuses'].append(dict(phase='B', status=status()))
        third = generate('model')          # SWITCH B→A on both ranks
        report['statuses'].append(dict(phase='A2', status=status()))
        report['generations'] = [first, second, third]
        report['deterministic'] = first['text'] == third['text']
        report['completed'] = bool(first['tokens'] and second['tokens'] and third['tokens'])
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    main()
