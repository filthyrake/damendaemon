"""Bounded A30 Q4 32K capacity gate with Q8_0 KV cache (#535 gate 2).

The exact #70 contract (pinned Qwen3.8-27B 1d4bf0f2, Q4 compensated, context
32768, input 32256 raw ids / output 512, C1 batch 1, prefill chunk 127, no
graphs) retried with DAMENDAEMON_HYBRID_KV=q8_0. Same preflight floors, same
request limits, one retry cycle: the gate either passes inside the #70 memory
envelope or fails with the allocator evidence retained. Capacity only; no
quality or performance qualification; the eager full-attention decode path
(quantized KV bypasses the fused dense attention kernel) is recorded.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.experiments import hybrid_q4_32k_gate as base
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

KV_MODE = 'q8_0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    base.preflight()
    import torch
    prompt_ids = base.build_prompt_ids()
    digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=base.SNAPSHOT, quantization='q4_0'))))
    report: dict = dict(completed=False, gate='32K C1 capacity with Q8_0 KV (#535 gate 2)', model=base.SNAPSHOT,
                  context=base.CONTEXT, prompt_tokens=len(prompt_ids), prompt_digest=digest,
                  output_limit=base.OUTPUT_TOKENS, chunk=base.CHUNK, batch=1, quantization='q4_0',
                  kv_mode=KV_MODE, precision='compensated', graphs=False, device=base.A30_UUID,
                  torch=torch.__version__, rounds={}, statuses={})
    save(args.output, report)
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    environment['DAMENDAEMON_HYBRID_KV'] = KV_MODE
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.experiments.hybrid_q4_context_probe', 'serve',
             '--model-config', manifest_path, '--device', 'cuda', '--context', str(base.CONTEXT),
             '--no-graph', '--prefill-chunk-size', str(base.CHUNK), '--batch-size', '1',
             '--port', str(base.PORT), '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '600'],
            env=environment, stdout=open(args.output.with_suffix('.serve.log'), 'w'),
            stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{base.PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('serve exited before becoming healthy')
                time.sleep(10)
        else:
            raise RuntimeError('serve did not become healthy within 900 seconds')
        report['rounds']['warmup'] = base.generate(prompt_ids[:512], 1, socket_timeout=900)
        report['statuses']['after_warmup'] = base.status()
        save(args.output, report)
        if report['rounds']['warmup'].get('error'):
            raise RuntimeError(f"warmup failed: {report['rounds']['warmup']['error']}")
        report['rounds']['gate_32k'] = base.generate(prompt_ids, base.OUTPUT_TOKENS)
        report['statuses']['after_gate'] = base.status()
        gate = report['rounds']['gate_32k']
        usage = gate.get('usage') or {}
        report['completed'] = bool(
            gate.get('first_output_seconds') is not None
            and gate['first_output_seconds'] <= base.FIRST_OUTPUT_LIMIT
            and gate['total_seconds'] <= base.REQUEST_LIMIT
            and usage.get('completion_tokens') == base.OUTPUT_TOKENS
            and not gate.get('error'))
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
