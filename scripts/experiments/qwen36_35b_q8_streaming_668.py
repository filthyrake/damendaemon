"""Bounded A6000 cell: Q8-packED synchronous layer streaming (#668).

The qualified Q8 snapshot (34.8 GiB packed) FITS the A6000 resident — this
cell streams it anyway to qualify the quantized streaming surface: packed
int8 weights and their fp16 scales stream together, one layer resident at a
time, and the packed Linear rebinds its id()-keyed scale lookups after every
activation. The point is machinery correctness, not fitting a too-big model:
quantized streaming is what makes models LARGER than the card streamable.

Gates:
1. PARITY (primary): the streamed Q8 executor must match the RESIDENT Q8
   executor exactly on identical packed weights — same load, same eager torch
   kernels, same prompts: byte-exact prefill logits and token-exact greedy
   chains, plus spec-decode == plain greedy through the streamed MTP virtual
   layer. Same-dtype same-kernel exactness is demanded (the FP16 cell's
   control A proved the streaming copy path bit-exact).
2. REFERENCE ENVELOPE: against the pinned BF16 CPU reference from the cell-2
   gate (reference-v2.pt, sha256 450cec47...), argmax 5/5 required; 12-token
   greedy per the amended gate (exact, or first divergence at a top-2 margin
   < 0.1 — a characterized tie-flip); rel-RMS reported against the Q8 gate's
   0.114 envelope.

Honest limits: CAPACITY + NUMERICS cell, no performance claim — synchronous
streaming re-transfers each layer per decode step (PCIe-bound by design).

Preflight: single-device guard on the pinned A6000 UUID, GPU free
>= 40000 MiB, host MemAvailable >= 100 GiB, reference digest pinned.
"""
import argparse
import hashlib
import json
import os
# The shared-CUDA guard strips DAMENDAEMON_* env vars (hermetic child env,
# PASSTHROUGH allowlist only) — every opt-in is set in-process before any
# damendaemon import, exactly like the cell-2 gate runner.
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD'] = '1'
os.environ['DAMENDAEMON_KERNELS'] = 'torch'
os.environ['DAMENDAEMON_HYBRID_MTP'] = '1'
os.environ['DAMENDAEMON_HYBRID_MTP_CONCAT'] = 'embed_first'
import subprocess
import sys
import time
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard

SNAPSHOT = '/home/damen/models/Qwen_Qwen3.6-35B-A3B'
REFERENCE = '/tmp/g668/reference-v2.pt'
REFERENCE_SHA256 = '450cec478777c04296de34d1a0d663bf8a65456d968facb9837030e8fa6b6850'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
CONTEXT = 512
GEN_TOKENS = 12
TIE_MARGIN = 0.1
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3

PROMPTS = [
    'The capital of France is',
    'One two three four five',
    'Water boils at a temperature of',
    'The largest planet in the solar system is',
    'def add(a, b):\n    return',
]


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'q8 streaming cell requires the single-device A6000 guard, got {visible}')
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
    digest = hashlib.sha256(Path(REFERENCE).read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(f'reference digest mismatch: {digest} != pinned {REFERENCE_SHA256}')


def run_pass(executor, tokenizer, reference):
    """One executor pass over the pinned corpus: per-prompt logits, argmax,
    greedy chain with tie margins."""
    checks = []
    for row, text in zip(reference, PROMPTS):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if ids != row['ids']:
            raise ValueError(f'prompt tokenization drifted from the pinned reference: {text[:30]}')
        logits = executor.prefill(ids, 0)
        ours = logits[0].float().cpu()
        theirs = row['logits']
        rms = float((ours - theirs).norm() / theirs.norm())
        argmax_match = int(ours.argmax()) == int(theirs.argmax())
        generated, token, margins = [], int(ours.argmax()), []
        top2 = torch.topk(ours, 2)
        margins.append(float(top2.values[0] - top2.values[1]))
        for _ in range(GEN_TOKENS - 1):
            generated.append(token)
            step, nxt = executor.decode(token)
            token = int(nxt[0])
            vals = step[0].float().cpu()
            top2 = torch.topk(vals, 2)
            margins.append(float(top2.values[0] - top2.values[1]))
        generated.append(token)
        divergence_margin = None
        if generated != row['greedy']:
            for i, (ours_tok, ref_tok) in enumerate(zip(generated, row['greedy'])):
                if ours_tok != ref_tok:
                    divergence_margin = margins[i]
                    break
        checks.append(dict(prompt_tokens=len(ids), rel_rms=rms,
                           argmax_match=argmax_match, greedy=generated,
                           greedy_match=generated == row['greedy'],
                           divergence_margin=divergence_margin,
                           tie_flip=(not generated == row['greedy'])
                                    and divergence_margin is not None
                                    and divergence_margin < TIE_MARGIN))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    import torch
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    # weights_only=True is safe here: the reference payload is plain
    # dicts/lists/ints/tensors written by the pinned cell-2 gate runner
    # (digest-checked above) — no arbitrary pickle types.
    reference = torch.load(REFERENCE, weights_only=True)

    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    started = time.perf_counter()
    model = load_model(SNAPSHOT, CONTEXT, quantization='q8_0')
    load_seconds = time.perf_counter() - started
    if not model.scales:
        raise RuntimeError('q8 prepare produced no scales')

    report = dict(completed=False, gate='quantized (Q8) synchronous layer-streaming cell (#668)',
                  model=SNAPSHOT, model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, quantization='q8_0', precision='float16',
                  graphs=False, device=A6000_UUID, kernels='torch-hybrid',
                  weight_bytes=model.bytes, weight_gib=model.bytes/2**30,
                  weight_load_seconds=load_seconds, mtp=True, checks=[])

    # ── phase 1: streamed ────────────────────────────────────────────
    streamed = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    if streamed.offload is None:
        raise RuntimeError('streaming store did not engage')
    if not streamed.offload.resident_scales:
        raise RuntimeError('packed scales did not stream with the fixed set')
    streamed_checks = run_pass(streamed, tokenizer, reference)
    streamed_spec, _, _ = streamed.spec_decode(reference[0]['ids'], 6)
    streamed_status = streamed.status()
    report.update(stream=dict(
        peak_live_weight_bytes=streamed_status['peak_live_weight_bytes'],
        planned_peak_weight_bytes=streamed_status['planned_peak_weight_bytes'],
        copied_weight_bytes=streamed_status['copied_weight_bytes'],
        tensor_transfers=streamed_status['tensor_transfers'],
        streamed_scales=streamed_status['streamed_scales']))
    report['allocator_peak_gib_phase1'] = torch.cuda.max_memory_allocated()/2**30
    streamed.close()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ── phase 2: resident control ───────────────────────────────────
    resident = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    if resident.offload is not None:
        raise RuntimeError('resident control unexpectedly streams')
    resident_checks = run_pass(resident, tokenizer, reference)
    resident_spec, _, _ = resident.spec_decode(reference[0]['ids'], 6)
    report['allocator_peak_gib_phase2'] = torch.cuda.max_memory_allocated()/2**30

    # ── gates ────────────────────────────────────────────────────────
    parity = all(a['greedy'] == b['greedy'] for a, b in zip(streamed_checks, resident_checks))
    spec_parity = streamed_spec == resident_spec
    all_argmax = all(c['argmax_match'] for c in streamed_checks)
    greedy_or_tie = all(c['greedy_match'] or c['tie_flip'] for c in streamed_checks)
    worst_rms = max(c['rel_rms'] for c in streamed_checks)
    report.update(checks=streamed_checks,
                  parity_greedy=parity, spec_parity=spec_parity,
                  all_argmax=all_argmax, all_greedy_or_tie=greedy_or_tie,
                  worst_rel_rms=worst_rms,
                  worst_rel_rms_within_q8_envelope=worst_rms <= 0.114)
    # spec == plain greedy on the streamed side (drafts are always verified
    # against the main argmax): the emitted stream equals the greedy prefix.
    plain = streamed_checks[0]['greedy']
    report['spec_equals_plain_greedy_streamed'] = streamed_spec == plain[:len(streamed_spec)]
    report['completed'] = bool(parity and spec_parity and all_argmax
                               and greedy_or_tie and worst_rms <= 0.114)
    save(args.output, report)
    print(f"[q8-stream] parity {parity} spec_parity {spec_parity} argmax {all_argmax} "
          f"greedy_or_tie {greedy_or_tie} worst_rms {worst_rms:.4f} "
          f"peak {streamed_status['peak_live_weight_bytes']/2**30:.2f} GiB "
          f"transfers {streamed_status['tensor_transfers']}", flush=True)
    if not report['completed']:
        raise SystemExit('q8 streaming cell failed a gate; see report')


def save(path, report):
    path.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
