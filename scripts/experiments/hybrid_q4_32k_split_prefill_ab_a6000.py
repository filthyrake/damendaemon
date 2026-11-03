"""Split prefill admission A/B: the aggregate-prefill bound (#690).

The chunked scheduler admitted the whole batch into prefill at once, so
clustered arrivals serialized every first output onto the aggregate
prefill time (C4 first outputs 502-528 s at 4 slots; C5 projects ~630 s
but is POLICY-REFUSED — the compensated Q4 precision admits at most four
slots on this card, a measured memory bound). This runner measures the
split-admission fix at the qualified shape: 4 simultaneous 32K requests
on the 27B spec serve (the #685 production C4 shape), one guarded
invocation per phase:

- phase split   : DAMENDAEMON_PREFILL_ADMISSION=1 (one prefill lane)
- phase legacy  : admission 0 (batch-wide round-robin, the control —
                  expected to reproduce the aggregate bound: every
                  slot's first output waits on all four prefills)

Queue timeout is 900 s so the capacity surface (queue wait) is measured
rather than conflated with rejection; the SLO gates are POST-ADMISSION
per the #59 contract: client-observed first output minus the server-
reported queue_seconds (both recorded per slot).

Gates (declared on #690 before launch): both phases 4/4 complete 512/512
with zero errors and no queue-timeout rejections; phase split passes
per-slot post-admission first output <= 600 s and earliest-slot
client-observed first output <= 300 s (a single prefill's time — the
aggregate bound put EVERY slot at ~500 s); phase legacy confirms the
aggregate bound (every first output >= 450 s client-observed, spread
within 90 s — all slots waiting on all prefills); slot-0 digest must
reproduce the pinned C-series corpus digest; per-slot digests compared
across phases and reported (admission order changes scheduling, not
computation).

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB.
"""
import argparse
import hashlib
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

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
PORT = 8148
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
CHUNK = 1024
SLOTS = 4
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200
PINNED_PROMPT_DIGEST = ('31836e963e26cd642684d7bce9c7781323768'
                        '00375c2b09917612a5d398be727')


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'gate requires the single-device A6000 guard, got {visible}')
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
    read_timeout = socket_timeout if socket_timeout is not None else REQUEST_LIMIT + 600
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
                if time.monotonic() - started > REQUEST_LIMIT + 600:
                    raise RuntimeError(f'slot {slot} exceeded the wall limit')
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if first_output is None and (event.get('choices') or [{}])[0].get('text'):
                    first_output = time.monotonic() - started
                if event.get('usage'):
                    usage = event['usage']
                if event.get('damendaemon_timing'):
                    usage = usage or {}
                    usage['queue_seconds'] = event['damendaemon_timing'].get('queue_seconds')
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    chunks.append(choices[0]['text'])
    except Exception as failure:
        error = failure
    text = ''.join(chunks)
    results[slot] = dict(
        first_output_seconds=first_output, total_seconds=time.monotonic() - started,
        queue_seconds=(usage or {}).get('queue_seconds'),
        post_admission_first_output=(first_output - (usage or {}).get('queue_seconds')
                                     if first_output is not None and (usage or {}).get('queue_seconds') is not None
                                     else None),
        usage=usage, output_chars=len(text),
        output_ids_digest=hashlib.sha256(json.dumps(text).encode()).hexdigest(),
        error=str(error) if error else None)
    if text:
        results[slot]['output_preview'] = text[:80]
    print(f"[slot{slot}] first {first_output and round(first_output,1)}s "
          f"(queued {(usage or {}).get('queue_seconds') and round((usage or {})['queue_seconds'],1)}s) "
          f"total {round(results[slot]['total_seconds'],1)}s "
          f"tokens {(usage or {}).get('completion_tokens')} error={bool(error)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('split', 'legacy'), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    base = build_prompt_ids()
    prompts = [slot_prompt(base, slot) for slot in range(SLOTS)]
    digests = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]
    if digests[0] != PINNED_PROMPT_DIGEST:
        raise ValueError('slot-0 digest does not match the pinned C-series corpus; refusing to launch')
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate='#690 split prefill admission A/B (C4, 27B spec serve)',
                  phase=args.phase, model=SNAPSHOT,
                  model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS, output_limit=OUTPUT_TOKENS,
                  chunk=CHUNK, batch=SLOTS, quantization='q4_0', precision='compensated',
                  graphs=False, device=A6000_UUID, slot_prompt_digests=digests,
                  queue_timeout=900, torch=torch.__version__, rounds={}, statuses={})
    save(args.output, report)
    environment = dict(os.environ)
    environment['DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4'] = '1'
    environment['DAMENDAEMON_HYBRID_PRECISION'] = 'compensated'
    environment['DAMENDAEMON_HYBRID_MTP'] = '1'
    environment['DAMENDAEMON_PREFILL_ADMISSION'] = '1' if args.phase == 'split' else '0'
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.experiments.hybrid_q4_a6000_probe', 'serve',
             '--model-config', str(manifest_path), '--device', 'cuda', '--context', str(CONTEXT),
             '--no-graph', '--prefill-chunk-size', str(CHUNK),
             '--batch-size', str(SLOTS), '--port', str(PORT),
             '--gpu-models', '1', '--host-models', '1', '--host-gib', '40',
             '--queue-timeout', '900', '--queue-size', str(SLOTS)],
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
                    raise RuntimeError(f'serve exited {server.returncode} during warmup')
                time.sleep(5)
        else:
            raise RuntimeError('serve never became healthy')
        warmup = {}
        generate('warmup', prompts[0][:512], 4, warmup, socket_timeout=900)
        report['rounds']['warmup'] = dict(value=warmup.get('warmup'))
        report['statuses']['after_warmup'] = status()
        save(args.output, report)
        if warmup.get('warmup', {}).get('error'):
            raise RuntimeError(f"warmup failed: {warmup['warmup']['error']}")

        results = {}
        threads = [threading.Thread(target=generate,
                                    args=(slot, prompts[slot], OUTPUT_TOKENS, results))
                   for slot in range(SLOTS)]
        round_started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        report['rounds']['c4'] = dict(seconds=time.monotonic() - round_started,
                                      slots=results)
        report['statuses']['after_round'] = status()
        tokens_ok = all((results[slot].get('usage') or {}).get('completion_tokens') == OUTPUT_TOKENS
                        for slot in range(SLOTS))
        errors = [results[slot]['error'] for slot in range(SLOTS) if results[slot]['error']]
        if errors or not tokens_ok:
            raise RuntimeError(f'cell incomplete: errors={errors} tokens_ok={tokens_ok}')
        first = [results[slot]['first_output_seconds'] for slot in range(SLOTS)]
        post = [results[slot]['post_admission_first_output'] for slot in range(SLOTS)]
        totals = [results[slot]['total_seconds'] for slot in range(SLOTS)]
        report['first_output_seconds'] = first
        report['post_admission_first_output_seconds'] = post
        report['total_seconds'] = totals
        report['queue_seconds'] = [results[slot]['queue_seconds'] for slot in range(SLOTS)]
        if args.phase == 'split':
            report['completed'] = bool(
                all(value is not None and value <= FIRST_OUTPUT_LIMIT for value in post)
                and first[0] is not None and first[0] <= 300
                and all(value <= REQUEST_LIMIT + 600 for value in totals))
        else:
            # The control CONFIRMS the aggregate bound: every slot's first
            # output waits on all four prefills (>= 450 s, tight spread).
            report['aggregate_bound_confirmed'] = bool(
                all(value is not None and value >= 450 for value in first)
                and max(first) - min(first) <= 90)
            report['completed'] = report['aggregate_bound_confirmed']
        save(args.output, report)
        print(f"[{args.phase}] first={[round(v,1) if v else v for v in first]} "
              f"queued={[round(v,1) if v is not None else v for v in report['queue_seconds']]} "
              f"post={[round(v,1) if v is not None else v for v in post]} -> {report['completed']}", flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
    if not report['completed']:
        raise SystemExit('phase failed a gate; see report')


if __name__ == '__main__':
    main()
