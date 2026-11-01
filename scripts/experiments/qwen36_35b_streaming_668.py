"""Bounded A6000 streaming cell: run a too-big model on a smaller card (#668).

The pinned Qwen3.6-35B-A3B snapshot is 64.6 GiB as unquantized FP16 —
larger than the A6000's 44.5 GiB. This cell runs it ANYWAY through the
synchronous layer-streaming path (DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD=1):
host-resident weights, one layer's tensors resident on device at a time,
eager torch FP16, context within the streaming qualification bound.

Gates (accuracy/drift per the #23 convention, same as the cell-2 Q8 gate):
prefill last-token argmax agreement and 12-token greedy exactness against the
PINNED BF16 CPU reference from the cell-2 gate (reference-v2.pt, sha256
450cec478777c04296de34d1a0d663bf8a65456d968facb9837030e8fa6b6850) — the
same five prompts, same snapshot revision. FP16 streaming is expected to
sit inside the Q8 gate's envelope (worst rel-RMS 0.114, greedy exact);
the measured rel-RMS is reported, not gated, and the argmax/greedy gates
mirror the Q8 cell's.

Honest limits, stated up front: this is a CAPACITY + NUMERICS cell. No
performance claim — synchronous streaming re-transfers each layer per
decode step (PCIe-bound by design), and no warmed steady state is
claimed. The point is that the too-big model executes correctly through
the streaming path on hardware that cannot hold it.

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB (the 64.6 GiB FP16
weights are host-resident). Near-capacity configuration (guard reserve 0,
thermal opt-in 90 C) is required from the caller exactly as pre-declared.
"""
import argparse
import hashlib
import json
import os
# The shared-CUDA guard strips DAMENDAEMON_* env vars (hermetic child env,
# PASSTHROUGH allowlist only) — the streaming opt-in must be set in-process
# before any damendaemon import, exactly like the cell-2 gate runner's
# DAMENDAEMON_QUANTIZATION.
os.environ['DAMENDAEMON_EXPERIMENTAL_HYBRID_OFFLOAD'] = '1'
os.environ['DAMENDAEMON_KERNELS'] = 'torch'
import subprocess
import sys
import time
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save

SNAPSHOT = '/home/damen/models/Qwen_Qwen3.6-35B-A3B'
REFERENCE = '/tmp/g668/reference-v2.pt'
REFERENCE_SHA256 = '450cec478777c04296de34d1a0d663bf8a65456d968facb9837030e8fa6b6850'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
CONTEXT = 512
GEN_TOKENS = 12
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3

PROMPTS = [
    'The quick brown fox jumps over the lazy dog.',
    'A repository records every bounded qualification cell with evidence.',
    'Hybrid linear attention mixes with full attention every few layers.',
    'The measurement is the gate; the gate is not the measurement.',
    'Greedy decoding deterministically selects the argmax token at each step, which is precisely what verification requires.',
]


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'streaming cell requires the single-device A6000 guard, got {visible}')
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
    model = load_model(SNAPSHOT, CONTEXT, quantization='none')
    load_seconds = time.perf_counter() - started
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    if executor.offload is None:
        raise RuntimeError('streaming store did not engage')
    host_resident = all(t.device.type == 'cpu' for t in model.weights.values())

    report = dict(completed=False, gate='too-big-model streaming cell (#668 family extension)',
                  model=SNAPSHOT, model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, quantization='none', precision='float16',
                  graphs=False, device=A6000_UUID, kernels=executor.kernel_backend,
                  weight_bytes=model.bytes, weight_gib=model.bytes/2**30,
                  weight_load_seconds=load_seconds,
                  host_resident_weights=host_resident,
                  planned_peak_weight_bytes=executor.offload.layout.peak_bytes,
                  output_limit=GEN_TOKENS, prompts=len(PROMPTS),
                  reference=REFERENCE, reference_sha256=REFERENCE_SHA256,
                  q8_gate_reference='docs/evidence/nvidia/qwen36-35b-a3b-q8-a6000/ (PR #670)',
                  torch=torch.__version__, checks=[])
    save(args.output, report)

    try:
        for row, text in zip(reference, PROMPTS):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if ids != row['ids']:
                raise ValueError(f'prompt tokenization drifted from the pinned reference: {text[:30]}')
            t0 = time.perf_counter()
            logits = executor.prefill(ids, 0)
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - t0
            ours = logits[0].float().cpu()
            theirs = row['logits']
            rms = float((ours - theirs).norm() / theirs.norm())
            argmax_match = int(ours.argmax()) == int(theirs.argmax())
            generated, token = [], int(ours.argmax())
            decode_started = time.perf_counter()
            for _ in range(GEN_TOKENS - 1):
                generated.append(token)
                _, nxt = executor.decode(token)
                token = int(nxt[0])
            generated.append(token)
            decode_seconds = time.perf_counter() - decode_started
            greedy_match = generated == row['greedy']
            report['checks'].append(dict(
                prompt_tokens=len(ids), prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds, rel_rms=rms,
                argmax_match=argmax_match, greedy_match=greedy_match,
                ours_greedy=generated, ref_greedy=row['greedy']))
            save(args.output, report)
            print(f"[stream] {len(ids)}tok rms {rms:.4f} argmax {argmax_match} "
                  f"greedy {greedy_match} prefill {prefill_seconds:.1f}s "
                  f"decode {decode_seconds:.1f}s", flush=True)
        report['peak_live_weight_bytes'] = executor.offload.peak_live_bytes
        report['transferred_weight_bytes'] = executor.offload.copied_bytes
        report['tensor_transfers'] = executor.offload.transfers
        report['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['all_argmax'] = all(c['argmax_match'] for c in report['checks'])
        report['all_greedy'] = all(c['greedy_match'] for c in report['checks'])
        report['worst_rel_rms'] = max(c['rel_rms'] for c in report['checks'])
        report['completed'] = bool(report['all_argmax'] and report['all_greedy'])
        save(args.output, report)
        print(f"[stream] worst rel-RMS {report['worst_rel_rms']:.4f}; argmax {report['all_argmax']}; "
              f"greedy {report['all_greedy']}; peak live weights "
              f"{report['peak_live_weight_bytes']/2**30:.2f} GiB; transfers "
              f"{report['tensor_transfers']}", flush=True)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise
    finally:
        executor.offload.close()


if __name__ == '__main__':
    main()
