"""Bounded A6000 Q4 32K lifecycle cells (#68 follow-up, parent #60).

The remaining #68 gate-order items after the C-series (PR #659/#660):
cancellation and reuse, each sized and separately bounded before launch.
This runner executes ONE bounded cell per invocation, the same pinned
model/format/context/output as the passing C1 cell and the qualified
C2 level (2 slots), reusing the C-series corpus and the pinned slot-0
digest so every prompt is reproducible against prior evidence.

Cells (--cell):
- cancel-prefill: two concurrent 32K requests (the qualified C2 level);
  slot 0 disconnects 90 s after response-header admission (measured
  mid-decode on this stack — see the harness caveat below); slot 1
  must complete within the C1 per-request gates. After the round the
  allocator must show the cancelled slot's KV released (current returns
  to the warmup baseline), then a fresh 32K/64-token request must pass
  the gates — proving the slot pool is fully usable, not leaked.
  Harness caveat (measured, cell 4): urllib's urlopen returns only when
  the server sends response headers, which on this executor happens at
  first output (~308 s at batch 2), so cancel_after counts from header
  admission — a true mid-prefill disconnect would need a raw-socket
  client. Both cancel cells therefore measured mid-decode disconnects.
- cancel-decode: one 32K/512 request, client disconnects shortly after
  first streamed output (mid-decode); slot release asserted the same way
  via the follow-up request and allocator status.
- reuse-identical: the same 32K prompt twice sequentially — the repeat
  must reproduce the first completion byte-for-byte (greedy) with no
  allocator growth, recording whether the executor accelerates the
  repeat (prefix reuse) or re-prefills.
- reuse-suffix: fresh reference prompt P0, then P_tail (the same 30000-
  token prefix, a DIFFERENT final 2256 tokens) twice — the suffix change
  must change the completion (no stale replay of P0), and the repeat must
  reproduce P_tail byte-for-byte.

Per-request limits are the C1 gates (first streamed output <= 600 s,
complete request <= 1200 s, per request); outer guard 1800 s.
Cancellation, reuse and release behavior only: not quality or SLO
qualification.

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
PORT = 8131
CONTEXT = 32768
PROMPT_TOKENS = 32256
PREFIX_TOKENS = 30000
OUTPUT_TOKENS = 512
CHUNK = 127
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
FIRST_OUTPUT_LIMIT = 600
REQUEST_LIMIT = 1200
CANCEL_PREFILL_AFTER = 90
CANCEL_DECODE_HOLD = 5
PINNED_PROMPT_DIGEST = ('31836e963e26cd642684d7bce9c7781323768'
                        '00375c2b09917612a5d398be727')
# Allocator tolerance for "released back to baseline": one residual
# workspace/workspace-sized block, NOT a 4 GiB KV slot.
RELEASE_TOLERANCE_BYTES = 512 * 1024**2


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'lifecycle gate requires the single-device A6000 guard, got {visible}')
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


def status(endpoint='models/status'):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/{endpoint}', timeout=10) as response:
            return json.loads(response.read())
    except Exception as error:
        return dict(error=str(error))


def output_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def hard_disconnect(response):
    """Delegate to the shared deterministic-FIN helper (see
    scripts/experiments/http_disconnect.py for the full rationale)."""
    from scripts.experiments.http_disconnect import hard_disconnect as close_it
    close_it(response)


def request(slot, prompt_ids, max_tokens, results: dict, *, cancel_after=None,
            cancel_after_first_output=0):
    """One streamed completion. With cancel_after, the connection is closed
    that many seconds in (mid-prefill); with cancel_after_first_output,
    it is closed that many seconds after the first streamed content."""
    body = json.dumps(dict(model='qwen38', prompt=prompt_ids, max_tokens=max_tokens,
                           temperature=0, stream=True,
                           stream_options={'include_usage': True})).encode()
    request_ = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                                      headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first_output = None
    disconnected = None
    admitted = None
    chunks = []
    usage = None
    error = None
    response = None
    try:
        response = urllib.request.urlopen(request_, timeout=REQUEST_LIMIT + 90)
        admitted = time.monotonic() - started
        if cancel_after is not None:
            time.sleep(cancel_after)
            disconnected = time.monotonic() - started
            response.close()
            hard_disconnect(response)
        else:
            for line in response:
                if time.monotonic() - started > REQUEST_LIMIT:
                    raise RuntimeError(f'slot {slot} exceeded the 1200 second limit')
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
                choices = event.get('choices') or []
                if choices and choices[0].get('text'):
                    chunks.append(choices[0]['text'])
                if (cancel_after_first_output and first_output is not None
                        and time.monotonic() - started >= first_output + cancel_after_first_output):
                    disconnected = time.monotonic() - started
                    response.close()
                    hard_disconnect(response)
                    break
    except Exception as failure:
        error = failure
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    text = ''.join(chunks)
    results[slot] = dict(
        first_output_seconds=first_output, total_seconds=time.monotonic() - started,
        admitted_seconds=admitted, disconnected_seconds=disconnected, usage=usage,
        output_chars=len(text), output_digest=output_digest(text),
        error=str(error) if error else None)
    if text:
        results[slot]['output_preview'] = text[:80]


def release_probe(name, prompt_ids, results: dict, attempts=12, spacing=5.0):
    """A tiny request sent right after a client disconnect: if the cancelled
    slot was actually released, this is admitted within seconds; if the
    zombie request holds the slot, it queues behind the survivor instead.

    Retry-bounded (#664): the probe fires within milliseconds of the
    disconnect, but release itself takes ~one decode step (the EOF is
    observed at the next generation event). A single instant probe races
    that window and reads a 503 that says nothing about the hold; bounded
    retries distinguish prompt release (admitted within a few seconds)
    from a zombie (all attempts rejected for the full decode)."""
    import time as _time
    for attempt in range(attempts):
        attempt_results = {}
        request(name, prompt_ids, 1, attempt_results)
        result = attempt_results.get(name)
        results[name] = result
        if result is None or (result.get('error') is None
                              and result.get('first_output_seconds') is not None):
            return
        _time.sleep(spacing)


def followup_ok(result):
    return (result.get('first_output_seconds') is not None
            and result['first_output_seconds'] <= FIRST_OUTPUT_LIMIT
            and result['total_seconds'] <= REQUEST_LIMIT
            and not result.get('error'))


def full_ok(result, max_tokens):
    return (followup_ok(result)
            and (result.get('usage') or {}).get('completion_tokens') == max_tokens
            and (result.get('usage') or {}).get('prompt_tokens') == PROMPT_TOKENS)


def allocated(state):
    model = (state.get('models') or [{}])[0] if isinstance(state, dict) else {}
    return state.get('cuda_allocated_bytes') if isinstance(state, dict) else None


class Timeline:
    """Polls server status during a round for cancel-detection evidence."""
    def __init__(self):
        self.events = []
        self.stop = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._poll, daemon=True)
        thread.start()

    def _poll(self):
        while not self.stop.wait(15):
            self.events.append(dict(t=round(time.monotonic(), 1),
                                    models=status(), scheduler=status('scheduler/status')))

    def finish(self):
        self.stop.set()
        time.sleep(0.1)
        return self.events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=('cancel-prefill', 'cancel-decode',
                                            'reuse-identical', 'reuse-suffix'), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    base = build_prompt_ids()
    prompts = {slot: slot_prompt(base, slot) for slot in (0, 1)}
    digests = {slot: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
               for slot, ids in prompts.items()}
    if digests[0] != PINNED_PROMPT_DIGEST:
        raise ValueError(f'slot-0 digest {digests[0]} does not match the pinned '
                         f'C1 corpus {PINNED_PROMPT_DIGEST}; refusing to launch')
    # reuse-suffix: identical 30000-token prefix, a different final 2256 tokens.
    tail = (base[1:1 + PROMPT_TOKENS - PREFIX_TOKENS])
    suffix_prompt = prompts[0][:PREFIX_TOKENS] + tail
    suffix_digest = hashlib.sha256(json.dumps(suffix_prompt).encode()).hexdigest()
    if suffix_prompt == prompts[0]:
        raise ValueError('suffix control collided with the pinned prompt')
    batch = 2 if args.cell == 'cancel-prefill' else 1
    manifest_path = args.output.with_suffix('.manifest.json')
    save(manifest_path, dict(version=1, models=dict(qwen38=dict(path=SNAPSHOT, quantization='q4_0'))))
    report = dict(completed=False, gate=f'32K {args.cell} lifecycle',
                  executor_rescope='A6000 (issue #68, 2026-10-28)',
                  model=SNAPSHOT, model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS, output_limit=OUTPUT_TOKENS,
                  chunk=CHUNK, batch=batch, quantization='q4_0', precision='compensated',
                  graphs=False, device=A6000_UUID, slot_prompt_digests=digests,
                  suffix_prompt_digest=suffix_digest,
                  c1_reference='docs/evidence/nvidia/a6000-q4-32k-68/ gate-attempt5-PASSED.json',
                  c2_reference='docs/evidence/nvidia/a6000-q4-32k-68/concurrency/ gate-c2.json',
                  torch=torch.__version__, rounds={}, statuses={}, timeline=[])
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
             '--batch-size', str(batch), '--port', str(PORT),
             '--gpu-models', '1', '--host-models', '1',
             '--host-gib', '40', '--queue-timeout', '600',
             '--queue-size', str(batch)],
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
        warmup = {}
        request('warmup', prompts[0][:512], 1, warmup)
        report['rounds']['warmup'] = dict(value=warmup.get('warmup'))
        baseline = status()
        report['statuses']['after_warmup'] = baseline
        report['baseline_allocated_bytes'] = allocated(baseline)
        save(args.output, report)
        if warmup.get('warmup', {}).get('error'):
            raise RuntimeError(f"warmup failed: {warmup['warmup']['error']}")

        if args.cell in ('reuse-identical', 'reuse-suffix'):
            sequence = ([0, 0] if args.cell == 'reuse-identical'
                         else [0, 'suffix', 'suffix'])
            expectations = ([0, 0] if args.cell == 'reuse-identical'
                            else [0, 'suffix', 'suffix'])
            for index, which in enumerate(sequence):
                result = {}
                prompt = prompts[0] if which == 0 else suffix_prompt
                request(0, prompt, OUTPUT_TOKENS, result)
                report['rounds'][f'request_{index + 1}'] = dict(
                    variant=which, value=result.get(0))
                report['statuses'][f'after_request_{index + 1}'] = status()
                save(args.output, report)
            r1, r2 = (report['rounds']['request_1']['value'],
                      report['rounds']['request_2']['value'])
            r3 = (report['rounds']['request_3']['value']
                  if 'request_3' in report['rounds'] else None)
            rounds_ok = all(full_ok(r, OUTPUT_TOKENS)
                           for r in ((r1, r2, r3) if r3 is not None else (r1, r2)))
            if args.cell == 'reuse-identical':
                checks = dict(both_within_gates=rounds_ok,
                              identical_output=r1.get('output_digest') == r2.get('output_digest'),
                              no_allocator_growth=(report['statuses']['after_request_2']
                                                   .get('cuda_allocated_bytes', 0) <=
                                                   report['statuses']['after_request_1']
                                                   .get('cuda_allocated_bytes', 0) + RELEASE_TOLERANCE_BYTES))
                report['verdict'] = checks
                report['completed'] = all(checks.values())
                report['repeat_acceleration'] = dict(
                    first_output_ratio=(r2.get('first_output_seconds') or 0) /
                                        (r1.get('first_output_seconds') or 1),
                    total_ratio=(r2.get('total_seconds') or 0) / (r1.get('total_seconds') or 1))
            else:
                checks = dict(all_within_gates=rounds_ok,
                              suffix_changed_output=(r1.get('output_digest') != r2.get('output_digest')),
                              suffix_repeat_identical=r2.get('output_digest') == r3.get('output_digest'))
                report['verdict'] = checks
                report['completed'] = all(checks.values())
        else:
            timeline = Timeline()
            timeline.start()
            if args.cell == 'cancel-prefill':
                results = {}
                threads = [threading.Thread(
                    target=request,
                    args=(0, prompts[0], OUTPUT_TOKENS, results),
                    kwargs=dict(cancel_after=CANCEL_PREFILL_AFTER), daemon=True),
                    threading.Thread(
                    target=request, args=(1, prompts[1], OUTPUT_TOKENS, results),
                    daemon=True)]
                for thread in threads:
                    thread.start()
                threads[0].join()  # cancel fires at ~90 s
                # Early-release probe while the survivor is still decoding:
                # an early probe completion proves the slot freed at
                # disconnect detection; a queued probe means a zombie hold.
                probe = {}
                release_probe('probe', prompts[0][:512], probe)
                report['rounds']['release_probe'] = dict(value=probe.get('probe'))
                save(args.output, report)
                threads[1].join()
                report['rounds']['cancel_round'] = dict(slots=results)
                report['statuses']['after_cancel_round'] = status()
                save(args.output, report)
                timeline_events = timeline.finish()
                cancelled, survivor = results.get(0), results.get(1)
                probe_result = probe.get('probe') or {}
                release = (report['statuses']['after_cancel_round'].get('cuda_allocated_bytes')
                           is not None
                           and report['statuses']['after_cancel_round']['cuda_allocated_bytes']
                           <= (report['baseline_allocated_bytes'] or 0) + RELEASE_TOLERANCE_BYTES)
                checks = dict(disconnect_recorded=bool(cancelled and cancelled.get('disconnected_seconds')),
                              survivor_within_gates=bool(survivor and full_ok(survivor, OUTPUT_TOKENS)),
                              early_release_probe=bool(
                                  probe_result.get('first_output_seconds') is not None
                                  and probe_result['first_output_seconds'] < 300
                                  and not probe_result.get('error')),
                              cancelled_kv_released=release)
                report['verdict'] = checks
                # The follow-up proves the slot pool is fully usable after the cancel.
                followup = {}
                request(0, prompts[1], 64, followup)
                report['rounds']['followup_after_cancel'] = dict(value=followup.get(0))
                report['statuses']['after_followup'] = status()
                save(args.output, report)
                checks['followup_within_gates'] = followup_ok(followup.get(0) or {})
                report['verdict'] = checks
                report['completed'] = all(checks.values())
                report['timeline'] = timeline_events
            else:  # cancel-decode
                results = {}
                request(0, prompts[0], OUTPUT_TOKENS, results,
                        cancel_after_first_output=CANCEL_DECODE_HOLD)
                # The disconnect lands mid-decode; a probe admitted within
                # seconds proves early release, a ~240 s wait means the
                # zombie completed first (batch 1).
                probe = {}
                release_probe('probe', prompts[0][:512], probe)
                report['rounds']['cancel_round'] = dict(slots=results)
                report['rounds']['release_probe'] = dict(value=probe.get('probe'))
                report['statuses']['after_cancel_round'] = status()
                save(args.output, report)
                timeline_events = timeline.finish()
                cancelled = results.get(0)
                probe_result = probe.get('probe') or {}
                release = (report['statuses']['after_cancel_round'].get('cuda_allocated_bytes')
                           is not None
                           and report['statuses']['after_cancel_round']['cuda_allocated_bytes']
                           <= (report['baseline_allocated_bytes'] or 0) + RELEASE_TOLERANCE_BYTES)
                checks = dict(disconnect_after_first_output=bool(
                                  cancelled and cancelled.get('disconnected_seconds')
                                  and cancelled.get('first_output_seconds')
                                  and cancelled['disconnected_seconds'] >
                                  cancelled['first_output_seconds']),
                              partial_output_recorded=bool(cancelled and cancelled.get('output_chars')),
                              early_release_probe=bool(
                                  probe_result.get('first_output_seconds') is not None
                                  and probe_result['first_output_seconds'] < 120
                                  and not probe_result.get('error')),
                              cancelled_kv_released=release)
                followup = {}
                request(0, prompts[0], 64, followup)
                report['rounds']['followup_after_cancel'] = dict(value=followup.get(0))
                report['statuses']['after_followup'] = status()
                save(args.output, report)
                checks['followup_within_gates'] = followup_ok(followup.get(0) or {})
                report['verdict'] = checks
                report['completed'] = all(checks.values())
                report['timeline'] = timeline_events
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
