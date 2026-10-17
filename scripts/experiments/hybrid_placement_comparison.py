"""A30 placement comparison driver (#497): one placement per guarded run.

Starts the damendaemon serve CLI for the pinned Qwen3.8-27B snapshot in the
requested placement, waits for loopback readiness, drives fixed prompts at
batch 1 and 4, records outputs and wall-clock goodput, and shuts the server
down. Raw controls only: the cross-placement comparison and any placement
decision happen after both runs' reports exist.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

MODEL = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/'
         'snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
PROMPTS = [
    'What is 17 * 23? Answer with one number.',
    'Name the capital of France in one word.',
    'Count from one to five with commas.',
    'What color is a banana? One word.',
]
PORT = 8123
BATCHES = (1, 4)


def health(timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return True
        except Exception:
            time.sleep(2)
    return False


def generate(prompt, max_tokens=32):
    body = json.dumps(dict(model='qwen38', prompt=prompt, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.loads(response.read())
    return dict(text=payload['choices'][0]['text'], tokens=payload['usage']['completion_tokens'],
                seconds=time.monotonic()-started)


def batch_round(prompts):
    results = [None]*len(prompts)
    started = time.monotonic()
    def drive(index):
        results[index] = generate(prompts[index])
    threads = [threading.Thread(target=drive, args=(index,)) for index in range(len(prompts))]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    seconds = time.monotonic()-started
    total = sum(entry['tokens'] for entry in results)
    return dict(prompts=prompts, results=results, wall_seconds=seconds,
                completion_tokens=total, tokens_per_second=total/seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--placement', choices=('hybrid', 'resident'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report = dict(completed=False, placement=args.placement, model=MODEL,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  context=512, batches=list(BATCHES), rounds={})
    save(args.output, report)
    environment = dict(os.environ)
    server = None
    try:
        if args.placement == 'hybrid':
            environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
            environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
        server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.experiments.hybrid_q4_context_probe', 'serve',
             '--model', f'qwen38={MODEL}', '--device', 'cuda', '--context', '512',
             '--no-graph', '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '24'],
            env=environment, stdout=open(args.output.with_suffix('.serve.log'), 'w'),
            stderr=subprocess.STDOUT)
        if not health(600):
            raise RuntimeError('serve did not become healthy within 600 seconds')
        report['ready_seconds'] = None
        for batch in BATCHES:
            prompts = PROMPTS*batch
            rounds = []
            for _ in range(2):
                rounds.append(batch_round(prompts))
            best = max(rounds, key=lambda round_: round_['tokens_per_second'])
            report['rounds'][f'batch{batch}'] = dict(rounds=rounds,
                                                     best_tokens_per_second=best['tokens_per_second'])
        report['completed'] = True
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
