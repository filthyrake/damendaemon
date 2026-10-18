"""B580 dense long-context gate (#63 gate 5): dense Qwen3 Q8 beyond 2048 tokens.

Downloads nothing; expects the pinned dense Qwen3-0.6B snapshot at
MODEL_DIR (revision c1899de289a04d12100db370d81485cdf75e47ca). Serves it twice
with Q8_0 packed weights — round A at context 16384 batch 1 (needle retrieval
inside a ~6K-token prompt, requested twice for greedy determinism), round B at
context 4096 batch 4 (concurrent distinct prompts). Records XPU allocator
counters against the 2 GiB display reserve. Capability only: no performance
claims (shared desktop); this is not an independent-reference logit gate.
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

MODEL_DIR = '/home/damen/models/Qwen3-0.6B-c1899de'
PORT = 8127
LONG_CONTEXT = 16384
CONCURRENT_CONTEXT = 4096
NEEDLE = 'ZEPHYR-7391'
CONCURRENT_PROMPTS = ['What is 17 * 23? Answer with one number.',
                      'Name the capital of France in one word.',
                      'Count from one to five with commas.',
                      'What color is a banana? One word.']


def environment():
    values = dict(os.environ)
    values.update(DAMENDAEMON_DEVICE='xpu', DAMENDAEMON_EXPERIMENTAL_XPU='1',
                  ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                  # The fixed upstream Level Zero loader is required for every
                  # XPU process; see docs/evidence/intel/b580-bringup-63/.
                  LD_LIBRARY_PATH='/home/damen/b580-dev/ze-loader/lib64:'
                                  + os.environ.get('LD_LIBRARY_PATH', ''))
    return values


def needle_prompt(fill_paragraphs=240):
    filler = ('The survey team recorded routine measurements of terrain, '
              'wind speed, and supply levels. No anomalies were noted. ')
    body = ''.join(filler for _ in range(fill_paragraphs))
    midpoint = len(body) // 2
    body = (body[:midpoint]
            + f' At exactly noon the operator logged one restricted access code: {NEEDLE}. '
            + body[midpoint:])
    return body + f'\n\nQuestion: what is the restricted access code logged at noon? Answer with only the code.'


def health(deadline):
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                if json.loads(response.read()).get('status') == 'ok':
                    return True
        except Exception:
            time.sleep(2)
    return False


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def generate(prompt, max_tokens=16):
    body = json.dumps(dict(model='qwen3', prompt=prompt, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:160]}') from None
    return dict(text=payload['choices'][0]['text'],
                tokens=payload['usage']['completion_tokens'],
                prompt_tokens=payload['usage'].get('prompt_tokens'),
                seconds=time.monotonic() - started)


def serve_round(context, batch, log_path, drive):
    manifest_path = log_path.with_name(log_path.name + '.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen3=dict(path=MODEL_DIR, quantization='q8_0'))))
    server = subprocess.Popen(
        [sys.executable, '-B', '-m', 'damendaemon', 'serve', '--model-config', str(manifest_path),
         '--device', 'xpu', '--context', str(context), '--batch-size', str(batch),
         '--no-graph', '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '300'],
        env=environment(), stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    try:
        if not health(time.monotonic() + 900):
            raise RuntimeError(f'context {context} serve did not become healthy within 900 seconds')
        before = status()
        result = drive()
        after = status()
        return dict(context=context, batch=batch, before=before, after=after, **result)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report: dict = dict(completed=False, gate='dense Qwen3 Q8 long context on XPU',
                        model=MODEL_DIR, revision='c1899de289a04d12100db370d81485cdf75e47ca',
                        torch=torch.__version__, rounds={})
    save(args.output, report)
    try:
        prompt = needle_prompt()
        def drive_long():
            first = generate(prompt)
            second = generate(prompt)
            return dict(first=first, second=second,
                        deterministic=first['text'] == second['text'],
                        needle_found=NEEDLE in first['text'])
        def drive_concurrent():
            import threading
            concurrent, errors = [], []
            def worker(value):
                try:
                    concurrent.append(generate(value, max_tokens=24))
                except Exception as failure:
                    errors.append(str(failure))
            threads = [threading.Thread(target=worker, args=(value,)) for value in CONCURRENT_PROMPTS]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            if errors:
                raise RuntimeError(f'concurrent generation failures: {errors}')
            return dict(concurrent=concurrent)
        report['rounds']['long_16k'] = serve_round(
            LONG_CONTEXT, 1, args.output.with_suffix('.serve-long.log'), drive_long)
        save(args.output, report)
        report['rounds']['concurrent_4k_batch4'] = serve_round(
            CONCURRENT_CONTEXT, 4, args.output.with_suffix('.serve-concurrent.log'), drive_concurrent)
        long_round = report['rounds']['long_16k']
        concurrent_round = report['rounds']['concurrent_4k_batch4']
        report['prompt_tokens'] = long_round['first'].get('prompt_tokens')
        report['completed'] = bool(long_round['needle_found'] and long_round['deterministic']
                                   and len(concurrent_round['concurrent']) == len(CONCURRENT_PROMPTS)
                                   and all(entry.get('tokens') for entry in concurrent_round['concurrent']))
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
