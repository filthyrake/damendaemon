"""Bounded A6000 MTP speculative-decode gate for Qwen3.8-27B Q4 32K (#68).

The C4 closure measured 32K Q4 decode-bound at ~0.55 tok/s/slot packed-4
(~1.03 single-slot) with the chunk lever exhausted — the named remaining
levers are decode-side (graphs, speculative decode, split admission #59).
This cell brings the existing MTP self-speculative machinery to the C-series
checkpoint family: the 27B carries a DENSE draft head (mtp.layers.0.mlp.
{gate,up,down}_proj — no router/experts), so the transcription grows a dense
FFN branch (structural, mirroring the main checkpoint's dense layers; the
virtual index N dispatches through _mlp_half unchanged).

The gate measures the two properties that matter, per the #668 MTP precedent:

1. EXACTNESS (the pass gate): the MTP speculative stream must equal the
   plain greedy stream token-for-token on every prompt, under BOTH concat
   orders — spec_decode is exact by construction, so any divergence is a
   defect in the dense transcription.
2. ACCEPTANCE (the measurement): accepted drafts / main steps at 32K on the
   pinned C-series corpus (the decode-bound regime). The correct concat
   order shows materially higher acceptance; the asymmetry pins the
   semantics empirically for this checkpoint family.

No performance claim: shared-card wall-clock, no warmed steady state; the
acceptance rate is draft-quality evidence. The follow-on concurrency cell
(if acceptance qualifies it) is separately declared work.

Shape: THREE separately guarded phase launches (the 27B dense decode at
~1.0 tok/s makes the three phases not fit one 1800 s guard): plain
reference (MTP dropped), then spec under each concat order; then a local
verdict phase compares digests. Each guarded phase: fresh process, pinned
snapshot/context/quantization (q4_0 compensated, kernels triton — the
qualified Q4 path), five short prompts (12 greedy tokens) plus the 32,256-
token C-series corpus prompt (512 greedy tokens).

Preflight: single-device guard on the pinned A6000 UUID, fresh GPU
headroom >= 40000 MiB, host MemAvailable >= 100 GiB. Near-capacity
configuration (guard reserve 0, thermal opt-in 90 C) is required from the
caller exactly as pre-declared on #68 for this card.
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

SNAPSHOT = '/home/damen/models/Qwen3.8-27B-1d4bf0f2'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'
CONTEXT = 32768
LONG_TOKENS = 32256
LONG_OUTPUT = 512
SHORT_OUTPUT = 12
GPU_FLOOR_MIB = 40000
HOST_FLOOR_BYTES = 100 * 1024**3
PHASE_ENV = {'DAMENDAEMON_QUANTIZATION': 'q4_0',
             'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4': '1',
             'DAMENDAEMON_HYBRID_PRECISION': 'compensated',
             'DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT': str(CONTEXT),
             'DAMENDAEMON_HYBRID_MTP': '1'}
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


def corpus_ids(count):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    paragraph = ('The quick brown fox jumps over the lazy dog while the repository '
                 'records every bounded qualification cell with evidence and pins. ')
    ids = tokenizer.encode(paragraph).ids
    copies = 1
    while len(ids) < 2 * count:
        copies *= 2
        ids = tokenizer.encode(paragraph * copies).ids
    return ids[:count]


def encode_shorts():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(SNAPSHOT) / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return [tokenizer.encode(p).ids for p in SHORT_PROMPTS]


def phase(args):
    """One GPU phase (invoked under its own guard): plain or spec (order)."""
    # The guard's hermetic child env strips DAMENDAEMON_*; set in-process
    # before any damendaemon import touches the policy modules.
    os.environ.update(PHASE_ENV)
    if args.phase == 'plain':
        os.environ['DAMENDAEMON_HYBRID_MTP_DROP'] = '1'
    else:
        os.environ['DAMENDAEMON_HYBRID_MTP_CONCAT'] = args.order
    import torch
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor

    started = time.perf_counter()
    model = load_model(SNAPSHOT, CONTEXT, quantization='q4_0')
    load_seconds = time.perf_counter() - started
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='triton')
    record = dict(phase=args.phase, order=args.order, model=SNAPSHOT,
                  model_revision=REVISION, context=CONTEXT, quantization='q4_0',
                  precision='compensated', graphs=False, device=A6000_UUID,
                  kernels=executor.kernel_backend,
                  weight_gib=model.bytes/2**30, weight_load_seconds=load_seconds,
                  mtp_loaded=executor.mtp_layer is not None,
                  mtp_layer=executor.mtp_layer, prompts=[], torch=torch.__version__)
    if args.phase == 'spec' and executor.mtp_layer is None:
        raise ValueError('spec phase loaded without the MTP head')

    def run_one(name, ids, count, seconds):
        began = time.perf_counter()
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
                    main_steps=steps, accepted=accepted,
                    seconds=round(time.perf_counter() - began, 1))
        record['prompts'].append(entry)
        save(args.output, record)
        detail = f'accepted {accepted}/{steps}' if accepted is not None else 'plain'
        print(f"[{args.phase}/{args.order or '-'}] {name}: {len(output)} tokens, {detail}", flush=True)

    shorts = encode_shorts()
    for index, ids in enumerate(shorts):
        run_one(f'short{index}', ids, SHORT_OUTPUT, None)
    run_one('long32k', corpus_ids(LONG_TOKENS), LONG_OUTPUT, None)
    record['allocator_peak_gib'] = torch.cuda.max_memory_allocated()/2**30
    record['completed'] = True
    save(args.output, record)
    print(f"[{args.phase}] completed, peak {record['allocator_peak_gib']:.2f} GiB", flush=True)


def verdict(args):
    """Local digest comparison across the three phase reports (no GPU)."""
    paths = {name: args.output.with_name(f'{args.output.stem}-{name}.json')
             for name in ('plain', 'spec_embed_first', 'spec_hidden_first')}
    for name, path in paths.items():
        if not path.exists():
            raise ValueError(f'missing phase report: {path}')
    phases = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if not all(p.get('completed') for p in phases.values()):
        raise ValueError('a phase report is incomplete')
    plain = {p['name']: p for p in phases['plain']['prompts']}
    report = dict(completed=False, gate='dense MTP speculative decode (#68 decode lever)',
                  model=SNAPSHOT, model_revision=REVISION, context=CONTEXT,
                  quantization='q4_0', precision='compensated', device=A6000_UUID,
                  exactness={}, acceptance={}, phases={})
    for spec_name in ('spec_embed_first', 'spec_hidden_first'):
        spec = phases[spec_name]
        report['exactness'][spec_name] = {
            p['name']: p['output_ids_digest'] == plain[p['name']]['output_ids_digest']
            for p in spec['prompts']}
        long_entry = next(p for p in spec['prompts'] if p['name'] == 'long32k')
        report['acceptance'][spec_name] = dict(
            accepted=long_entry['accepted'], main_steps=long_entry['main_steps'],
            rate=(long_entry['accepted'] / long_entry['main_steps'])
            if long_entry['main_steps'] else None)
    for name, p in phases.items():
        report['phases'][name] = dict(
            completed=p['completed'], mtp_loaded=p.get('mtp_loaded'),
            peak_gib=p.get('allocator_peak_gib'),
            prompts={q['name']: dict(tokens=q['output_tokens'], seconds=q['seconds'])
                     for q in p['prompts']})
    report['completed'] = all(all(v.values()) for v in report['exactness'].values())
    save(args.output, report)
    print(f"[verdict] exactness {json.dumps(report['exactness'])} -> "
          f"completed={report['completed']}", flush=True)
    print(f"[verdict] acceptance {json.dumps(report['acceptance'])}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('plain', 'spec', 'verdict'), required=True)
    parser.add_argument('--order', choices=('embed_first', 'hidden_first'))
    args = parser.parse_args()
    if args.phase == 'verdict':
        verdict(args)
        return
    if args.phase == 'spec' and not args.order:
        raise SystemExit('spec phase requires --order')
    if args.phase == 'verdict':
        return
    require_guard()
    if args.output.exists():
        raise ValueError('new output path required')
    preflight()
    phase(args)


if __name__ == '__main__':
    main()
