"""Bounded A6000 Q4 32K capacity-only HTTP gate (#68 re-scope, parent #60).

The SAME pinned workload as #70's A30 attempt 5 (2026-10-17), re-scoped to
the desktop RTX A6000 (44.47 GiB) after the 3090→A6000 swap: the A30 cell
measured a definitive ~260 MiB-short OOM at 21.97/23.60 GiB, so the workload
is known-bounded near ~22 GiB; this is not an unchanged retry — the
executor hardware is new and adequate, per the re-scope recorded on #68.

Pinned Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, Q4
compensated, context 32768, input 32256 raw token ids (deterministic corpus;
the prompt digest is verified against the A30 attempt's pinned value
31836e963e26cd642684d7bce9c778132376800375c2b09917612a5d398be727 at
runtime and the run refuses to launch on mismatch) / output 512, C1
(batch 1), prefill chunk 127, no graphs. Preflight requires the
single-device guard on the A6000 UUID with fresh GPU headroom >= 40000 MiB
and host MemAvailable >= 100 GiB. A small warmup request activates the
model before the measured 32K request. Limits: first streamed output
<= 600 s, entire request <= 1200 s, outer guard 1800 s; stop on any
failure. Reports prompt digest, first-output and total seconds, usage
token counts, and allocator current/peak from /v1/models/status.
Capacity only: this is not quality or SLO qualification.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8128
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
CHUNK = 127
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200
# The A30 attempt-5 corpus digest; the deterministic builder must reproduce
# it exactly or this run is NOT the pinned workload and must not launch.
PINNED_PROMPT_DIGEST = ('31836e963e26cd642684d7bce9c7781323768'
                        '00375c2b09917612a5d398be727')


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'32K gate requires the single-device A6000 guard, got {visible}')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=30).stdout
    free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in query.strip().splitlines()}
    headroom = free.get(A6000_UUID)
    if headroom is None or headroom < GPU_FLOOR_MIB:
        raise ValueError(f'A6000 headroom {headroom} MiB is below the {GPU_FLOOR_MIB} MiB floor')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None or available < HOST_FLOOR_BYTES:
        raise ValueError(f'host MemAvailable {available} bytes is below the {HOST_FLOOR_BYTES} floor')


def build_prompt_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    if not ids:
        raise ValueError('prompt paragraph tokenized to zero ids')
    copies = 1
    # BPE merges across repetition boundaries: never assume per-copy counts
    # scale; measure the encoded length each round.
    while len(ids) < PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    if len(ids) < PROMPT_TOKENS:
        raise ValueError('prompt text did not tokenize to the required length')
    return ids[:PROMPT_TOKENS]


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def generate(prompt_ids, max_tokens, *, socket_timeout=None):
    # The client socket timeout must not clip activation: only the measured
    # gate request carries the 600 s first-output contract.
    read_timeout = socket_timeout if socket_timeout is not None else FIRST_OUTPUT_LIMIT + 30
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    chunks = []
    usage = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=read_timeout) as response:
            for line in response:
                if time.monotonic() - started > REQUEST_LIMIT:
                    raise RuntimeError('32K request exceeded the 1200 second limit')
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if first_output is None:
                    first_output = time.monotonic() - started
                if event.get('usage'):
                    usage = event['usage']
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    chunks.append(choices[0]['text'])
    except Exception as failure:
        error = failure
    result = dict(max_tokens=max_tokens, first_output_seconds=first_output,
                  total_seconds=time.monotonic() - started, usage=usage,
                  output_chars=sum(len(text) for text in chunks), error=str(error) if error else None)
    if chunks:
        result['output_preview'] = ''.join(chunks)[:120]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    prompt_ids = build_prompt_ids()
    digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
    if digest != PINNED_PROMPT_DIGEST:
        raise ValueError(f'corpus digest {digest} does not match the pinned '
                         f'A30 attempt-5 workload {PINNED_PROMPT_DIGEST}; refusing to launch')
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate='32K C1 capacity', executor_rescope='A6000 (issue #68, 2026-10-28)',
                  model=SNAPSHOT, model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                  context=CONTEXT, prompt_tokens=len(prompt_ids), prompt_digest=digest,
                  output_limit=OUTPUT_TOKENS, chunk=CHUNK, batch=1, quantization='q4_0',
                  precision='compensated', graphs=False, device=A6000_UUID,
                  prior_a30_attempt='docs/evidence/hybrid/a30-q4-32k-gate-70/ (attempt 5, OOM at 21.97/23.60 GiB)',
                  torch=torch.__version__, rounds={}, statuses={})
    save(args.output, report)
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.experiments.hybrid_q4_a6000_probe', 'serve',
             '--model-config', str(manifest_path), '--device', 'cuda', '--context', str(CONTEXT),
             '--no-graph', '--prefill-chunk-size', str(CHUNK), '--batch-size', '1',
             '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '600'],
            env=environment, stdout=open(args.output.with_suffix('.serve.log'), 'w'),
            stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
                    if json.loads(response.read()).get('status') == 'ok':
                        break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError('serve exited before becoming healthy')
                time.sleep(10)
        else:
            raise RuntimeError('serve did not become healthy within 900 seconds')
        # Activation is a fixed cost inside the first request (A6000 Q8
        # measured 61.9 s; the A30 measured ~616 s). Warm up so the measured
        # 32K request carries only its own prefill and decode; the warmup is
        # recorded but is not the gate cell.
        report['rounds']['warmup'] = generate(prompt_ids[:512], 1, socket_timeout=900)
        report['statuses']['after_warmup'] = status()
        save(args.output, report)
        if report['rounds']['warmup'].get('error'):
            raise RuntimeError(f"warmup failed: {report['rounds']['warmup']['error']}")
        report['rounds']['gate_32k'] = generate(prompt_ids, OUTPUT_TOKENS)
        report['statuses']['after_gate'] = status()
        gate = report['rounds']['gate_32k']
        usage = gate.get('usage') or {}
        report['completed'] = bool(
            gate.get('first_output_seconds') is not None
            and gate['first_output_seconds'] <= FIRST_OUTPUT_LIMIT
            and gate['total_seconds'] <= REQUEST_LIMIT
            and usage.get('completion_tokens') == OUTPUT_TOKENS
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
