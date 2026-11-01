"""Bounded A6000 MTP speculative-decode gate for Qwen3.6-35B-A3B (#668 MTP).

Upstream Transformers drops mtp.* at load, so no eager inference reference
exists for the draft head; the semantics were transcribed structurally
(DeepSeek-V3 MTP as carried by the Qwen3.6 layout) with one genuine open
choice — the fc input concat order (embedding-first vs hidden-first), which
no upstream code pins. This gate measures the two properties that matter:

1. EXACTNESS (the pass gate): the MTP speculative stream must equal the
   plain greedy stream token-for-token on every prompt, under BOTH concat
   orders — spec_decode is exact by construction (every draft is verified
   against the main head), so any divergence is a defect.
2. ACCEPTANCE (the measurement): accepted drafts / main steps on the
   trained checkpoint. The correct concat order should show materially
   higher acceptance than the wrong one (~0 on random projections); the
   measured asymmetry pins the semantics empirically.

No performance claim: shared-card wall-clock, no warmed steady state; the
acceptance rate is draft-quality evidence, not a speedup number.

Shape: three sequential subprocess phases (each loads the 34.76 GiB q8
weights fresh — GPU memory cannot hold two executors): plain reference
(MTP dropped), then spec under each concat order. Prompts: the five cell-2
gate prompts (5-9 tokens, 12 greedy tokens each) plus one 32,256-token
C-series corpus prompt (512 greedy tokens) for acceptance statistics and
long-context exactness. Same pinned snapshot/context/quantization as the
cell-2 gate (PR #670).

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB. Near-capacity
configuration (guard reserve 0, thermal opt-in 90 C) is required from the
caller exactly as pre-declared on #668 for this card.
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
LONG_TOKENS = 32256
LONG_OUTPUT = 512
SHORT_OUTPUT = 12
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
SHORT_PROMPTS = [
    'The quick brown fox jumps over the lazy dog.',
    'A repository records every bounded qualification cell with evidence.',
    'Hybrid linear attention mixes with full attention every few layers.',
    'The measurement is the gate; the gate is not the measurement.',
    'Greedy decoding deterministically selects the argmax token at each step, which is precisely what verification requires.',
]


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A6000_UUID]:
        raise ValueError(f'MTP gate requires the single-device A6000 guard, got {visible}')
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


def build_long_ids():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < 2 * LONG_TOKENS:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:LONG_TOKENS]


def encode_shorts():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    return [tokenizer.encode(p).ids for p in SHORT_PROMPTS]


def phase(args):
    """One in-process phase (invoked as a subprocess by the driver): plain or
    spec (order); writes its JSON and exits."""
    import torch
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor

    started = time.perf_counter()
    model = load_model(SNAPSHOT, CONTEXT, quantization='q8_0')
    load_seconds = time.perf_counter() - started
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='triton')
    record = dict(phase=args.phase, order=args.order, model=SNAPSHOT,
                  model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, quantization='q8_0', precision='float16',
                  graphs=False, device=A6000_UUID, kernels=executor.kernel_backend,
                  weight_gib=model.bytes/2**30, weight_load_seconds=load_seconds,
                  mtp_loaded=executor.mtp_layer is not None,
                  prompts=[], torch=torch.__version__)
    shorts = encode_shorts()

    def run_one(name, ids, count):
        if args.phase == 'plain':
            logits = executor.prefill(ids, 0)
            token = int(logits[0].float().argmax())
            output = [token]
            steps = count - 1
            accepted = None
            for _ in range(count - 1):
                logits, next_ids = executor.decode(token)
                token = int(next_ids[0])
                output.append(token)
        else:
            output, accepted, steps = executor.spec_decode(ids, count, 0)
        digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
        entry = dict(name=name, prompt_tokens=len(ids), output_tokens=len(output),
                    output_ids_digest=digest, output_preview_ids=output[:12],
                    main_steps=steps, accepted=accepted)
        record['prompts'].append(entry)
        save(args.output, record)
        print(f"[{args.phase}/{args.order or '-'}] {name}: {len(output)} tokens, "
              f"accepted {accepted}/{steps}" if accepted is not None else
              f"[{args.phase}] {name}: {len(output)} tokens", flush=True)

    for index, ids in enumerate(shorts):
        run_one(f'short{index}', ids, SHORT_OUTPUT)
    run_one('long', build_long_ids(), LONG_OUTPUT)
    record['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
    record['completed'] = True
    save(args.output, record)
    print(f"[{args.phase}] completed, peak {record['allocator_peak_gib']:.2f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('driver', 'plain', 'spec'), default='driver')
    parser.add_argument('--order', choices=('embed_first', 'hidden_first'))
    args = parser.parse_args()
    if args.phase != 'driver':
        if args.phase == 'spec' and not args.order:
            raise SystemExit('spec phase requires --order')
        phase(args)
        return
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    report = dict(completed=False, gate='MTP speculative decode (#668 MTP cell)',
                  model=SNAPSHOT, model_revision='995ad96eacd98c81ed38be0c5b274b04031597b0',
                  context=CONTEXT, quantization='q8_0', device=A6000_UUID,
                  phases={}, torch=__import__('torch').__version__)
    save(args.output, report)
    base = dict(os.environ)
    base['DAMENDAEMON_QUANTIZATION'] = 'q8_0'
    base['DAMENDAEMON_HYBRID_MTP'] = '1'
    try:
        for name, env in [('plain', {**base, 'DAMENDAEMON_HYBRID_MTP_DROP': '1'}),
                          ('spec_embed_first', {**base, 'DAMENDAEMON_HYBRID_MTP_CONCAT': 'embed_first'}),
                          ('spec_hidden_first', {**base, 'DAMENDAEMON_HYBRID_MTP_CONCAT': 'hidden_first'})]:
            path = args.output.with_name(f'{args.output.stem}-{name}.json')
            if path.exists():
                raise ValueError(f'phase output exists: {path}')
            started = time.perf_counter()
            child = subprocess.run(
                [sys.executable, '-B', '-m', 'scripts.experiments.qwen36_35b_mtp_gate_668',
                 '--output', str(path), '--phase', 'plain' if name == 'plain' else 'spec',
                 *([] if name == 'plain' else ['--order',
                     'embed_first' if name.endswith('embed_first') else 'hidden_first'])],
                env=env, cwd=str(Path(__file__).resolve().parents[2]),
                stdout=open(path.with_suffix('.log'), 'w'), stderr=subprocess.STDOUT,
                timeout=1500)
            report['phases'][name] = dict(exit_code=child.returncode,
                                          seconds=time.perf_counter() - started,
                                          output=path.name)
            save(args.output, report)
            if child.returncode != 0:
                raise RuntimeError(f'phase {name} failed (exit {child.returncode}); see {path.with_suffix(".log")}')
        # Verdict: exactness of both spec phases against plain, per prompt.
        plain = json.loads((args.output.with_name(f'{args.output.stem}-plain.json')).read_text())
        by_name = {p['name']: p for p in plain['prompts']}
        exact = {}
        for spec_name in ('spec_embed_first', 'spec_hidden_first'):
            spec = json.loads((args.output.with_name(f'{args.output.stem}-{spec_name}.json')).read_text())
            exact[spec_name] = {
                p['name']: p['output_ids_digest'] == by_name[p['name']]['output_ids_digest']
                for p in spec['prompts']}
        report['exactness'] = exact
        report['completed'] = all(all(v.values()) for v in exact.values())
        save(args.output, report)
        print(f"[gate] exactness {json.dumps(exact)} -> completed={report['completed']}", flush=True)
    except BaseException as error:
        report.update(completed=False, error=str(error))
        save(args.output, report)
        raise


if __name__ == '__main__':
    main()
