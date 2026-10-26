"""#606 gate-3 live cell: populated-slot handoff on the A30 -> A30+A2 pair.

One guarded process, in-process split pipeline (stage 0 = layers [0,12) on
the A30, stage 1 = [12,24) on the A2), Qwen3.5-0.8B unquantized, context 512.

Phases:
  1. Sender: a single-device HybridExecutor on the A30 prefills the pinned
     64-token prompt; the populated slot is exported (prompt-only snapshot);
     the sender then continues its own greedy chain on the live slot.
  2. Handoff: the exported payload installs into the fresh split pipeline
     (HybridPipeline.install_handoff); the pipeline continues greedy from
     the sender's first sampled token — the disaggregated-prefill contract.
  3. Control: a second split pipeline recomputes the same prompt and
     continues greedy.

Gate: the three 12-token chains are token-for-token identical, the sender's
slot releases cleanly after the acknowledged handoff, and handoff
bytes/duration land in the report. Correctness cell only — no throughput
claim (shared guarded GPUs).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

SNAPSHOT = ('/home/damen/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/'
            'snapshots/2fc06364715b967f1860aea9cf38778875588b17')
A30 = 'GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a'
A2 = 'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283'
PROMPT = list(range(16, 16 + 64))
GREEDY_TOKENS = 12
HEADROOM_FLOOR_MIB = 6000
RESULT = Path('/tmp/g606/handoff-cell.json')


def gpu_state(query):
    out = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30).stdout
    return {row.split(',')[0].strip(): row.split(',')[1].strip() for row in out.strip().splitlines()}


def preflight():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [A30, A2]:
        raise SystemExit(f'requires the two-device guard with the A30 first, got {visible}')
    free = {uuid: int(value) for uuid, value in gpu_state('uuid,memory.free').items()}
    for uuid in visible:
        if free.get(uuid, 0) < HEADROOM_FLOOR_MIB:
            raise SystemExit(f'{uuid} headroom {free.get(uuid)} MiB below {HEADROOM_FLOOR_MIB} MiB floor')


def build_stage(model, start, end, device_index):
    from damendaemon.hybrid_stage import HybridLayerStage
    stage = HybridLayerStage(model, start, end, max_batch=2, kernels='torch')
    device = torch.device('cuda', device_index)
    for group in ('w', 'kv', 'conv', 'state'):
        for key, tensor in list(getattr(stage, group).items()):
            getattr(stage, group)[key] = tensor.to(device)
    for name in ('cos', 'sin', 'arange'):
        if hasattr(stage, name):
            setattr(stage, name, getattr(stage, name).to(device))
    return stage


def sender_chain(sender, prompt, steps, first):
    tokens = [first]
    for _ in range(steps - 1):
        _, next_ids = sender.decode(tokens[-1])
        tokens.append(int(next_ids[0]))
    return tokens


def pipeline_chain(pipeline, first, steps):
    tokens = [first]
    for _ in range(steps - 1):
        _, next_ids = pipeline.decode_batch([tokens[-1]], [0])
        tokens.append(int(next_ids[0]))
    return tokens


def control_chain(pipeline, prompt, steps):
    tokens = [int(pipeline.prefill(list(prompt)).argmax(-1)[0])]
    for _ in range(steps - 1):
        _, next_ids = pipeline.decode_batch([tokens[-1]], [0])
        tokens.append(int(next_ids[0]))
    return tokens


def main():
    preflight()
    from damendaemon.models import load
    from damendaemon.hybrid import HybridExecutor
    from damendaemon.hybrid_pipeline import HybridPipeline
    from damendaemon.slot_handoff import export_slot, release_slot
    started = time.perf_counter()
    model_sender = load(SNAPSHOT, 512, None)
    model_stages = load(SNAPSHOT, 512, None)
    layers = model_stages.config['num_hidden_layers']
    split = layers // 2
    sender = HybridExecutor(model_sender, use_graph=False, max_batch=2, kernels='torch')
    pipeline = HybridPipeline([build_stage(model_stages, 0, split, 0),
                               build_stage(model_stages, split, layers, 1)])
    control = HybridPipeline([build_stage(model_stages, 0, split, 0),
                              build_stage(model_stages, split, layers, 1)])

    prefill_started = time.perf_counter()
    logits = sender.prefill(PROMPT)
    prefill_seconds = time.perf_counter() - prefill_started
    first = int(logits.argmax(-1)[0])
    # Export the prompt-only snapshot BEFORE the sender's own continuation
    # advances the live slot (export clones; the chain cannot touch it).
    payload = export_slot(sender, 0)
    live = sender_chain(sender, PROMPT, GREEDY_TOKENS, first)
    reports = pipeline.install_handoff(payload, slot=0, generation=1)
    handed = pipeline_chain(pipeline, first, GREEDY_TOKENS)
    recomputed = control_chain(control, PROMPT, GREEDY_TOKENS)
    release = release_slot(sender, 0)

    equal = handed == live == recomputed
    result = dict(
        gate='#606 gate 3 live handoff cell (in-process A30+A2 pair)',
        completed=equal,
        devices=dict(a30=A30, a2=A2, sender='cuda:0 (A30)',
                     pipeline=['cuda:0 stage [0,%d)' % split, 'cuda:1 stage [%d,%d)' % (split, layers)]),
        model=SNAPSHOT, layers=layers, split=split,
        prompt_tokens=len(PROMPT), greedy_tokens=GREEDY_TOKENS,
        first_sampled_token=first,
        chains=dict(sender=live, handoff=handed, recomputed=recomputed),
        sender_continuation_equals_handoff=handed == live,
        recomputed_control_equals_handoff=handed == recomputed,
        prefill_seconds=prefill_seconds,
        export=payload['telemetry'],
        install=reports,
        pipeline_status_handoffs=pipeline.status()['handoffs'],
        release=release,
        total_seconds=time.perf_counter() - started,
        gpu_used_after=gpu_state('uuid,memory.used'),
    )
    RESULT.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))
    if not equal:
        sys.exit(1)


if __name__ == '__main__':
    main()
