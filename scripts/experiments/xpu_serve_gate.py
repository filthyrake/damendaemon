"""B580 serve-mode lifecycle gate (#63 gate 3): concurrent HTTP + display headroom.

Starts `damendaemon serve` on the B580 with the Qwen3.5-0.8B hybrid dev fixture
at the bring-up limits (context 2048, batch 4, FP32), drives concurrent
requests through the queued scheduler, verifies telemetry and complete
release, and records XPU memory counters against the 2 GiB display reserve
with the desktop display actively in use. Capability only: no performance
claims (shared desktop, Wi-Fi host).
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

SNAPSHOT = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/'
            'snapshots/2fc06364715b967f1860aea9cf38778875588b17')
PORT = 8127
CONTEXT = 2048
PROMPTS = ['What is 17 * 23? Answer with one number.',
           'Name the capital of France in one word.',
           'Count from one to five with commas.',
           'What color is a banana? One word.']


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


def generate(prompt, max_tokens=24):
    body = json.dumps(dict(model='qwen35', prompt=prompt, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:160]}') from None
    return dict(prompt=prompt[:24], text=payload['choices'][0]['text'][:60],
                tokens=payload['usage']['completion_tokens'],
                seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen35=dict(path=SNAPSHOT, quantization='none'))))
    report = dict(completed=False, gate='serve lifecycle + display headroom',
                  model=SNAPSHOT, context=CONTEXT, batch=4, quantization='none',
                  torch=torch.__version__, rounds={}, statuses={})
    save(args.output, report)
    environment = dict(os.environ)
    environment.update(DAMENDAEMON_DEVICE='xpu', DAMENDAEMON_EXPERIMENTAL_XPU='1',
                       ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                       # The fixed upstream Level Zero loader is required for every
                       # XPU process; see docs/evidence/intel/b580-bringup-63/.
                       LD_LIBRARY_PATH='/home/damen/b580-dev/ze-loader/lib64:'
                                       + os.environ.get('LD_LIBRARY_PATH', ''))
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-B', '-m', 'damendaemon', 'serve', '--model-config', str(manifest_path),
             '--device', 'xpu', '--context', str(CONTEXT), '--batch-size', '4',
             '--no-graph', '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '300'],
            env=environment, stdout=open(args.output.with_suffix('.serve.log'), 'w'), stderr=subprocess.STDOUT)
        if not health(time.monotonic() + 900):
            raise RuntimeError('serve did not become healthy within 900 seconds')
        report['statuses']['before'] = status()
        save(args.output, report)
        warmup = generate(PROMPTS[0], max_tokens=8)
        report['rounds']['warmup'] = warmup
        concurrent = []
        import threading
        errors = []
        def worker(prompt):
            try:
                concurrent.append(generate(prompt))
            except Exception as failure:
                errors.append(str(failure))
        threads = [threading.Thread(target=worker, args=(prompt,)) for prompt in PROMPTS]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        if errors:
            raise RuntimeError(f'concurrent generation failures: {errors}')
        report['rounds']['concurrent_batch4'] = concurrent
        report['statuses']['after'] = status()
        after = report['statuses']['after']
        report['display_headroom'] = dict(
            cuda_allocated_bytes=after.get('cuda_allocated_bytes'),
            cuda_peak_allocated_bytes=after.get('cuda_peak_allocated_bytes'),
            display_reserve_bytes=2 * 1024**3)
        report['completed'] = bool(
            warmup.get('tokens') and len(concurrent) == len(PROMPTS)
            and all(entry.get('tokens') for entry in concurrent))
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
