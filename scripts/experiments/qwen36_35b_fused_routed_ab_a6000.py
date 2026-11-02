"""Fused routed dequant-matvec A/B for the Qwen3.6-35B-A3B MoE (#688).

_moe_half's q8_0 branch materialized every active expert slice through an
fp32 intermediate (dequantize -> .float()*scales -> fp16) before the
projections; at decode that is ~48 active experts x 40 layers x ~12 MB of
pure dequant traffic per step. This runner measures the existing
moe_kernels.routed_linear/combine path (gpt-oss/olmoe production surface,
register-resident dequant — the fp32 intermediate never exists) wired into
the qwen3_5_moe hot path, against the materialize path as control:

- ONE phase per invocation (--phase fused|materialize; the guard strips
  DAMENDAEMON_* so the phase env is set in-process before the import);
- same pinned snapshot/context/output as the #668/#686 cells:
  Qwen3.6-35B-A3B @ 995ad96e, q8_0 packed experts, context 32768,
  512 output tokens, Triton kernels, --no-graph, single-device A6000 guard;
- 8192-token prompts (not 32K) so six sequential prefills plus the decode
  phase fit one bounded guard window — decode at 8K positions is cheaper
  than the 32K C6 record's attention, shared by both phases; the A/B delta
  is the measurement, the C6 numbers are context, not the control;
- C6 geometry: 6 slots, deterministic per-slot rotations, whole prefill
  per slot, continuously batched greedy decode, 512 tokens per slot.

Gates (declared on #688 before launch):
- both phases: 6/6 slots, exact 512 tokens, finite logits, zero OOM;
- control sanity: materialize per-slot decode >= 1.0 tok/s (guards
  against both phases being equally broken);
- fused aggregate decode seconds strictly < materialize (the kernel must
  win; ratio reported);
- fused allocator peak <= materialize;
- output digests compared and REPORTED, not gated — the fused path sums
  routes in fp32 (combine) where the materialize path index_add'ed fp16
  contributions; divergence is characterized per the #23 discipline
  (cross-kernel exactness not demanded).

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB.
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
PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 512
SLOTS = 6
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('fused', 'materialize'), required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    # Set BEFORE importing damendaemon — the guard strips DAMENDAEMON_* from
    # the child env, so the phase is chosen in-process.
    os.environ['DAMENDAEMON_MOE_FUSED_ROUTED'] = '1' if args.phase == 'fused' else '0'
    import torch
    base = build_prompt_ids()
    prompts = [slot_prompt(base, slot) for slot in range(SLOTS)]
    digests = [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in prompts]

    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    t0 = time.perf_counter()
    model = load_model(SNAPSHOT, CONTEXT, quantization='q8_0')
    load_seconds = time.perf_counter() - t0
    executor = HybridExecutor(model, use_graph=False, max_batch=SLOTS, kernels='triton')

    report = dict(completed=False, gate='#688 fused routed dequant-matvec A/B (direct executor)',
                  phase=args.phase, model=SNAPSHOT,
                  model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, prompt_tokens=PROMPT_TOKENS,
                  slot_prompt_digests=digests, output_limit=OUTPUT_TOKENS,
                  batch=SLOTS, quantization='q8_0', precision='float16',
                  graphs=False, device=A6000_UUID, kernels=executor.kernel_backend,
                  weight_gib=model.bytes/2**30, weight_load_seconds=load_seconds,
                  kv_cache_bytes=executor.kv_cache_bytes,
                  torch=torch.__version__, phases={}, slots={})
    save(args.output, report)

    # Warmup: flush Triton JIT (including the routed kernels' tile variants)
    # before any measured phase, exactly as the #684 serve A/B did.
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

        tokens = {slot: [report['slots'][f'slot{slot}']['first_token']]
                  for slot in range(SLOTS)}
        current = {slot: tokens[slot][-1] for slot in range(SLOTS)}
        oom_events = 0
        decode_started = time.perf_counter()
        for step in range(OUTPUT_TOKENS - 1):
            logits, nxt = executor.decode_batch([current[s] for s in range(SLOTS)],
                                                 list(range(SLOTS)))
            for row, slot in enumerate(range(SLOTS)):
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
            aggregate_tokens_per_second=SLOTS*OUTPUT_TOKENS/decode_seconds,
            per_slot_tokens_per_second=OUTPUT_TOKENS/decode_seconds)

        for slot in range(SLOTS):
            entry = report['slots'][f'slot{slot}']
            entry['output_tokens'] = len(tokens[slot])
            entry['output_ids_digest'] = hashlib.sha256(
                json.dumps(tokens[slot]).encode()).hexdigest()
            entry['output_preview_ids'] = tokens[slot][:16]
            entry['passed'] = bool(entry['output_tokens'] == OUTPUT_TOKENS)
        report['phases']['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['phases']['allocator_current_gib'] = torch.cuda.memory_allocated()/2**30
        report['oom_count'] = oom_events
        report['completed'] = bool(all(
            report['slots'][f'slot{slot}']['passed'] for slot in range(SLOTS)))
        save(args.output, report)
        print(f"[ab] phase={args.phase}: prefill {sum(prefill_seconds):.1f}s decode "
              f"{decode_seconds:.1f}s ({report['phases']['decode']['per_slot_tokens_per_second']:.2f} "
              f"tok/s/slot) peak {report['phases']['allocator_peak_gib']:.2f} GiB "
              f"-> completed={report['completed']}", flush=True)
    except torch.cuda.OutOfMemoryError as error:
        report.update(completed=False, error=f'OOM: {error}', oom_count=oom_events+1)
        report['phases']['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        save(args.output, report)
        raise
    except BaseException as error:
        report.update(completed=False, error=str(error))
        report['phases']['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
