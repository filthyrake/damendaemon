"""Two-rank A30+A2 pair gate (#51): the qualified distributed-serve point.

Starts the two-rank Qwen2.5-0.5B greedy service (context 512, q4_0, batch 4)
across the guard's device pair, waits for loopback health, drives the same
greedy prompt twice (determinism), and records outputs and usage. Requires the
two-device guard: CUDA_VISIBLE_DEVICES is a comma pair, one rank per device.
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
PROMPT = 'What is 17 * 23? Answer with one number.'
PORT = 8124
PROMPT_IDS = [9707, 1879]  # recorded in the #218 manifest for 'Hello' style raw input


def health(deadline):
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return True
        except Exception:
            time.sleep(2)
    return False


def generate(prompt, max_tokens=16):
    body = json.dumps(dict(model='model', prompt=prompt, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:200]}') from None
    return dict(text=payload['choices'][0]['text'], tokens=payload['usage']['completion_tokens'],
                seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 2 or any(not uuid.startswith('GPU-') for uuid in visible):
        raise ValueError('the pair gate requires the two-device guard')
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report = dict(completed=False, model=SNAPSHOT, devices=visible,
                  torch=torch.__version__, quantization='q4_0', context=512, generations=[])
    save(args.output, report)
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-m', 'torchrun', '--nproc-per-node', '2', '--master-port', '29500',
             '-m', 'damendaemon', 'distributed-serve', SNAPSHOT,
             '--quantization', 'q4_0', '--context', '512', '--port', str(PORT),
             '--batch-size', '4', '--queue-timeout', '300', '--idle-timeout', '600'],
            stdout=open(args.output.with_suffix('.serve.log'), 'w'), stderr=subprocess.STDOUT)
        if not health(300):
            raise RuntimeError('two-rank service did not become healthy within 300 seconds')
        first = generate(PROMPT)
        second = generate(PROMPT)
        report['generations'] = [first, second]
        report['deterministic'] = first['text'] == second['text']
        report['completed'] = bool(first['tokens'] and second['tokens'])
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
