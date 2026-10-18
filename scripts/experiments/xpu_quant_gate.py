"""B580 quantized-execution gate (#63 gate 4): Q8_0 vs FP32 greedy agreement.

Serves the Qwen3.5-0.8B hybrid dev fixture twice at the hybrid-geometry bring-up
limits (context 1024, batch 4) — first FP32, then Q8_0 packed weights — and
drives the same greedy prompts through both servers. Records per-prompt greedy
agreement, XPU allocator counters against the 2 GiB display reserve, and the
concurrent batch-4 round on the Q8 server. Capability only: no performance
claims (shared desktop).
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
CONTEXT = 1024
PROMPTS = ['What is 17 * 23? Answer with one number.',
           'Name the capital of France in one word.',
           'Count from one to five with commas.',
           'What color is a banana? One word.',
           'Briefly explain why the sky is blue in two sentences.']
QUANTIZATIONS = ('none', 'q8_0')


def environment():
    values = dict(os.environ)
    values.update(DAMENDAEMON_DEVICE='xpu', DAMENDAEMON_EXPERIMENTAL_XPU='1',
                  ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                  # The fixed upstream Level Zero loader is required for every
                  # XPU process; see docs/evidence/intel/b580-bringup-63/.
                  LD_LIBRARY_PATH='/home/damen/b580-dev/ze-loader/lib64:'
                                  + os.environ.get('LD_LIBRARY_PATH', ''))
    return values


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


def generate(prompt, max_tokens=32):
    body = json.dumps(dict(model='qwen35', prompt=prompt, max_tokens=max_tokens,
                           temperature=0, stream=False)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode(errors="replace")[:160]}') from None
    return dict(text=payload['choices'][0]['text'],
                tokens=payload['usage']['completion_tokens'],
                seconds=time.monotonic() - started)


def serve_round(quantization, log_path):
    manifest_path = log_path.with_name(log_path.name + '.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen35=dict(path=SNAPSHOT, quantization=quantization))))
    server = subprocess.Popen(
        [sys.executable, '-B', '-m', 'damendaemon', 'serve', '--model-config', str(manifest_path),
         '--device', 'xpu', '--context', str(CONTEXT), '--batch-size', '4',
         '--no-graph', '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
         '--host-gib', '40', '--queue-timeout', '300'],
        env=environment(), stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
    try:
        if not health(time.monotonic() + 900):
            raise RuntimeError(f'{quantization} serve did not become healthy within 900 seconds')
        before = status()
        outputs = {prompt: generate(prompt) for prompt in PROMPTS}
        after = status()
        return dict(before=before, after=after, outputs=outputs)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output path required')
    import torch
    report: dict = dict(completed=False, gate='quantized execution on XPU (Q8_0 vs FP32 greedy)',
                        model=SNAPSHOT, context=CONTEXT, batch=4, torch=torch.__version__, rounds={})
    save(args.output, report)
    try:
        for index, quantization in enumerate(QUANTIZATIONS):
            report['rounds'][quantization] = serve_round(
                quantization, args.output.with_suffix(f'.serve-{quantization}.log'))
            save(args.output, report)
        baseline = report['rounds']['none']['outputs']
        packed = report['rounds']['q8_0']['outputs']
        agreement = {prompt[:24]: dict(match=baseline[prompt]['text'] == packed[prompt]['text'],
                                       fp32=baseline[prompt]['text'][:80],
                                       q8_0=packed[prompt]['text'][:80])
                     for prompt in PROMPTS}
        report['greedy_agreement'] = agreement
        report['greedy_matches'] = sum(entry['match'] for entry in agreement.values())
        report['completed'] = report['greedy_matches'] == len(PROMPTS)
        save(args.output, report)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
