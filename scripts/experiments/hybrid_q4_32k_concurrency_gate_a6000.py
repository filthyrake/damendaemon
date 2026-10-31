"""Bounded A6000 Q4 32K concurrency cells (#68 follow-up, parent #60).

Per #68's gate order, C1 success unlocks — does not assume — concurrency:
size and separately bound it before attempting C4. This runner executes
ONE bounded C{N} attempt per invocation (N = 2 or 4, batch/queue N), the
same pinned model/format/context/output as the passing C1 cell
(PR #659, docs/evidence/nvidia/a6000-q4-32k-68/), with N deterministic
per-slot prompt variants (rotations of the same pinned corpus — distinct
token id arrays per slot, digests recorded; slot 0 must reproduce the
pinned C1 digest exactly or the run refuses to launch). Per-request
limits are the C1 gates: first streamed output <= 600 s and complete
request <= 1200 s PER REQUEST; outer guard 1800 s. Reports per-request
first-output/total/usage, aggregate time, and allocator current/peak.
Capacity only: not cancellation, reuse, quality or SLO qualification.

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB. Near-capacity
configuration (guard reserve 0, thermal opt-in 90 C) is required from
the caller exactly as pre-declared on #68 for this card.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8130
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
CHUNK = 127
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200
PINNED_PROMPT_DIGEST = ('31836e963e26cd642684d7bce9c7781323768'
                        '00375c2b09917612a5d398be727')


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'concurrency gate requires the single-device A6000 guard, got {visible}')
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
    while len(ids) < 2 * PROMPT_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    if len(ids) < 2 * PROMPT_TOKENS:
        raise ValueError('prompt text did not tokenize to the required length')
    return ids


def slot_prompt(base, slot):
    """Deterministic per-slot rotation: distinct ids per slot, same corpus."""
    if slot == 0:
        ids = base[:PROMPT_TOKENS]
    else:
        ids = (base[slot:] + base[:slot])[:PROMPT_TOKENS]
    if ids == base[:PROMPT_TOKENS] and slot != 0:
        raise ValueError(f'slot {slot} rotation collided with slot 0')
    return ids


def status():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models/status', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def generate(slot, prompt_ids, max_tokens, results, *, socket_timeout=None):
    read_timeout = socket_timeout if socket_timeout is not None else REQUEST_LIMIT + 90
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
                    raise RuntimeError(f'slot {slot} exceeded the 1200 second limit')
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
    results[slot] = dict(
        first_output_seconds=first_output,
        total_seconds=time.monotonic() - started,
        usage=usage, output_chars=sum(len(text) for text in chunks),
        error=str(error) if error else None)
    if chunks:
        results[slot]['output_preview'] = ''.join(chunks)[:80]


def generate_one(slot, prompt_ids, max_tokens):
    """Sequential single-request helper (the warmup only)."""
    holder = {}
    generate(slot, prompt_ids, max_tokens, holder, socket_timeout=900)
    return holder[slot]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    base = build_prompt_ids()
    prompts = [slot_prompt(base, slot) for slot in range(args.concurrency)]
    digests = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
    # The slot-0 rotation must reproduce the pinned C1 corpus exactly.
    if digests[0] != PINNED_PROMPT_DIGEST:
        raise ValueError(f'slot-0 digest {digests[0]} does not match the pinned '
                         f'C1 corpus {PINNED_PROMPT_DIGEST}; refusing to launch')
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate=f'32K C{args.concurrency} concurrency',
                  executor_rescope='A6000 (issue #68, 2026-10-28)',
                  model=SNAPSHOT, model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS,
                  slot_prompt_digests=digests, output_limit=OUTPUT_TOKENS, chunk=CHUNK,
                  batch=args.concurrency, quantization='q4_0', precision='compensated',
                  graphs=False, device=A6000_UUID,
                  c1_reference='docs/evidence/nvidia/a6000-q4-32k-68/ gate-attempt5-PASSED.json',
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
             '--no-graph', '--prefill-chunk-size', str(CHUNK),
             '--batch-size', str(args.concurrency),
             '--port', str(PORT), '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '600',
             '--queue-size', str(args.concurrency)],
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
        warmup = generate_one(0, prompts[0][:512], 1)
        report['rounds']['warmup'] = dict(value=warmup)
        report['statuses']['after_warmup'] = status()
        save(args.output, report)
        if warmup.get('error'):
            raise RuntimeError(f"warmup failed: {warmup['error']}")
        # The measured C{N} cell: N concurrent 32K requests, one thread each.
        results = {}
        started = time.monotonic()
        threads = [threading.Thread(target=generate,
                                    args=(slot, prompts[slot], OUTPUT_TOKENS, results),
                                    daemon=True)
                   for slot in range(args.concurrency)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        report['rounds'][f'gate_32k_c{args.concurrency}'] = dict(
            slots=results, aggregate_seconds=time.monotonic() - started)
        report['statuses']['after_gate'] = status()
        slots_pass = all(
            slot.get('first_output_seconds') is not None
            and slot['first_output_seconds'] <= FIRST_OUTPUT_LIMIT
            and slot['total_seconds'] <= REQUEST_LIMIT
            and (slot.get('usage') or {}).get('completion_tokens') == OUTPUT_TOKENS
            and not slot.get('error')
            for slot in results.values())
        report['completed'] = bool(slots_pass and len(results) == args.concurrency)
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
