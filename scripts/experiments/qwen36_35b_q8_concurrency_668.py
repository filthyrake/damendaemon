"""Bounded A6000 Q8 32K concurrency cells for the Qwen3.6-35B-A3B MoE (#668).

The 27B dense cells (#68, docs/evidence/nvidia/a6000-q4-32k-68/) qualified
2 slots at 32K and measured C4 throughput-bound; the MoE hypothesis is that
3B-active decode re-opens the SLO window (#59/#60) that the dense model
could not reach. This runner executes ONE bounded C{N} attempt per
invocation (N = 1, 2 or 4 slots) against the direct hybrid executor
(the HTTP lifecycle is cell 3's own work):

- same pinned snapshot/context/output as the passing cell-2 Q8 gate
  (PR #670): Qwen3.6-35B-A3B @ 995ad96e, q8_0 packed experts, context
  32768, 512 output tokens, Triton kernels, --no-graph;
- N deterministic per-slot prompt rotations (distinct token id arrays per
  slot, digests recorded; slot 0 pinned across cells);
- whole-prefill per slot (chunked prefill is NOT admitted for this
  configuration — CUDA q8_0 fp16 — so each slot's 32,256-token prompt
  prefills in one forward, sequentially per slot exactly as prefill_batch
  composes);
- continuously batched decode (decode_batch over all live slots), greedy
  per slot, 512 tokens per slot total;
- per-slot gates inherited from the #68 C1 discipline: prefill
  <= 600 s, whole request (prefill + 512-token decode) <= 1200 s,
  exact token counts, finite logits.

Capacity + SLO scaling only: not cancellation, reuse, quality or HTTP
lifecycle qualification, and no wall-clock performance claim beyond the
gates (shared-GPU timings follow the repo's evidence discipline — the
measured numbers here are capacity/SLO feasibility, not benchmarks).

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB. Near-capacity
configuration (guard reserve 0, thermal opt-in 90 C) is required from
the caller exactly as pre-declared on #668 for this card.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen_Qwen3.6-35B-A3B'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
CONTEXT = 32768
PROMPT_TOKENS = 32256
OUTPUT_TOKENS = 512
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
PREFILL_LIMIT = 600
REQUEST_LIMIT = 1200


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    base = build_prompt_ids()
    prompts = [slot_prompt(base, slot) for slot in range(args.concurrency)]
    digests = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]

    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    t0 = time.perf_counter()
    model = load_model(SNAPSHOT, CONTEXT, quantization='q8_0')
    load_seconds = time.perf_counter() - t0
    executor = HybridExecutor(model, use_graph=False, max_batch=args.concurrency, kernels='triton')
    kv_bytes = executor.kv_cache_bytes

    report = dict(completed=False, gate=f'32K C{args.concurrency} concurrency (direct executor)',
                  model=SNAPSHOT, model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS,
                  slot_prompt_digests=digests, output_limit=OUTPUT_TOKENS,
                  batch=args.concurrency, quantization='q8_0', precision='float16',
                  graphs=False, device=A6000_UUID, kernels=executor.kernel_backend,
                  weight_gib=model.bytes/2**30, weight_load_seconds=load_seconds,
                  kv_cache_bytes=kv_bytes,
                  cell2_reference='docs/evidence/nvidia/qwen36-35b-a3b-q8-a6000/ (PR #670)',
                  torch=torch.__version__, phases={}, slots={})
    save(args.output, report)

    # Warmup: small prefill + a few batched decode steps. The first 32K
    # prefill additionally includes Triton JIT for the large scan shapes;
    # this is disclosed rather than excluded (the #68 cells' first measured
    # prefill carried the same property through serve activation).
    warm_start = time.perf_counter()
    logits = executor.prefill(prompts[0][:256], 0)
    token = int(logits[0].float().argmax())
    for _ in range(4):
        _, nxt = executor.decode_batch([token], [0])
        token = int(nxt[0])
    torch.cuda.synchronize()
    report['phases']['warmup'] = dict(seconds=time.perf_counter()-warm_start)
    report['phases']['allocator_after_warmup_gib'] = torch.cuda.max_memory_allocated()/2**30
    save(args.output, report)

    try:
        # Per-slot whole prefill, sequentially (prefill_batch composition).
        prefill_seconds = []
        for slot, prompt in enumerate(prompts):
            started = time.perf_counter()
            logits = executor.prefill(prompt, slot)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            prefill_seconds.append(seconds)
            if not torch.isfinite(logits[0].float()).all():
                raise ValueError(f'slot {slot} prefill logits not finite')
            report['slots'][f'slot{slot}'] = dict(
                prefill_seconds=seconds, first_token=int(logits[0].float().argmax()))
            report['phases']['prefill'] = dict(seconds=prefill_seconds,
                                               total=sum(prefill_seconds))
            save(args.output, report)

        # Continuously batched decode: one decode_batch step per token for
        # every live slot, greedy per slot.
        tokens = {slot: [report['slots'][f'slot{slot}']['first_token']]
                  for slot in range(args.concurrency)}
        current = {slot: tokens[slot][-1] for slot in range(args.concurrency)}
        decode_started = time.perf_counter()
        for step in range(OUTPUT_TOKENS - 1):
            logits, nxt = executor.decode_batch([current[s] for s in range(args.concurrency)],
                                                list(range(args.concurrency)))
            for row, slot in enumerate(range(args.concurrency)):
                tokens[slot].append(int(nxt[row]))
                current[slot] = int(nxt[row])
            if step % 128 == 127:
                torch.cuda.synchronize()
                report['phases']['decode_progress'] = dict(
                    steps=step+1, seconds=time.perf_counter()-decode_started)
                save(args.output, report)
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        report['phases']['decode'] = dict(
            seconds=decode_seconds, steps=OUTPUT_TOKENS - 1,
            aggregate_tokens_per_second=args.concurrency*OUTPUT_TOKENS/decode_seconds,
            per_slot_tokens_per_second=OUTPUT_TOKENS/decode_seconds)

        for slot in range(args.concurrency):
            entry = report['slots'][f'slot{slot}']
            entry['output_tokens'] = len(tokens[slot])
            entry['output_ids_digest'] = hashlib.sha256(
                json.dumps(tokens[slot]).encode()).hexdigest()
            entry['output_preview_ids'] = tokens[slot][:16]
            entry['request_seconds'] = entry['prefill_seconds'] + decode_seconds
            entry['passed'] = bool(entry['prefill_seconds'] <= PREFILL_LIMIT
                                   and entry['request_seconds'] <= REQUEST_LIMIT
                                   and entry['output_tokens'] == OUTPUT_TOKENS)
        report['phases']['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['phases']['allocator_current_gib'] = torch.cuda.memory_allocated()/2**30
        report['completed'] = bool(all(
            report['slots'][f'slot{slot}']['passed'] for slot in range(args.concurrency)))
        save(args.output, report)
        print(f"[cell] C{args.concurrency}: prefill {prefill_seconds} decode "
              f"{decode_seconds:.1f}s ({report['phases']['decode']['aggregate_tokens_per_second']:.1f} "
              f"tok/s aggregate) peak {report['phases']['allocator_peak_gib']:.2f} GiB "
              f"-> completed={report['completed']}", flush=True)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        report['phases']['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
