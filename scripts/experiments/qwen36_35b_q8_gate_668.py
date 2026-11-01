#!/usr/bin/env python3
"""#668 cell 2 — A6000 Q8 numerical gate for Qwen3.6-35B-A3B (qwen3_5_moe).

Two phases in one process (sequential; the 64.6 GiB BF16 reference and the
~35 GiB Q8 executor cannot coexist in 124 GB RAM):

  A. reference: HF transformers 5.20.1 Qwen3_5MoeForConditionalGeneration,
     BF16 on CPU, plain-text prompts (add_special_tokens=False), prefill
     logits + greedy continuation saved to /tmp;
  B. candidate: damendaemon q8_0 load on CUDA (Triton kernels, FP16
     compute), same prompts through the packed-MoE executor; rel-RMS of
     prefill logits, argmax agreement, greedy-token agreement over the
     continuation.

Convention: accuracy/drift per the #23 pattern (never byte-exactness across
quantized boundaries); the transformers 5.20.1 modeling file is the same
family as the 5.20.2-pinned routing reference. No performance claim.
Run guarded on the A6000.
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import gc
import json
import os
import sys
import time

import torch

PROMPTS = [
    'The capital of France is',
    'One two three four five',
    'Water boils at a temperature of',
    'The largest planet in the solar system is',
    'def add(a, b):\n    return',
]
GEN_TOKENS = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/damen/models/Qwen_Qwen3.6-35B-A3B')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--reference', default='/tmp/g668/reference.pt')
    parser.add_argument('--out', default='/tmp/g668/q8-gate.json')
    args = parser.parse_args()
    sys.path.insert(0, '/tmp/668b')
    os.environ.setdefault('DAMENDAEMON_QUANTIZATION', 'q8_0')

    # ── phase A: CPU BF16 reference ─────────────────────────────────
    if not os.path.exists(args.reference):
        print('[ref] loading HF reference (CPU BF16)...', flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map='cpu')
        reference = []
        t0 = time.perf_counter()
        for text in PROMPTS:
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            with torch.inference_mode():
                logits = model(torch.tensor([ids])).logits[0, -1].float()
                # generate's first new token IS the last-position argmax, so the
                # greedy stream is exactly the new tokens — no manual prepend
                # (prepending duplicates t0 and shifts every comparison by one)
                generated = model.generate(
                    torch.tensor([ids]), max_new_tokens=GEN_TOKENS,
                    do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.eos_token_id)[0, len(ids):].tolist()
            reference.append(dict(ids=ids, logits=logits.cpu(),
                                 greedy=generated))
        print(f'[ref] {len(reference)} prompts in {time.perf_counter()-t0:.0f}s',
              flush=True)
        torch.save(reference, args.reference)
        del model
        gc.collect()

    # ── phase B: damendaemon Q8 on CUDA ─────────────────────────────
    print('[cand] loading damendaemon q8_0 on CUDA...', flush=True)
    t0 = time.perf_counter()
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.model, args.context, quantization='q8_0')
    print(f'[cand] weights {model.bytes/2**30:.1f} GiB in '
          f'{time.perf_counter()-t0:.0f}s; cache={model.weight_cache_status}', flush=True)
    assert model.scales, 'q8 prepare produced no scales'
    packed = [name for name, t in model.weights.items()
              if name.endswith('.experts_gate_up')]
    assert all(model.weights[name].dtype is torch.int8 for name in packed), \
        'expert stacks are not packed'
    assert all(model.weights[name].dtype is not torch.int8
               for name in model.weights if name.endswith(('.router', '.shared_expert_gate'))), \
        'routing tensors must stay unpacked'
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='triton')
    print(f'[cand] executor up; backend {executor.kernel_backend}; '
          f'peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)

    reference = torch.load(args.reference, weights_only=False)
    checks = []
    for row in reference:
        logits = executor.prefill(row['ids'])
        ours = logits[0].float().cpu()
        theirs = row['logits']
        rms = float((ours-theirs).norm()/theirs.norm())
        argmax_match = int(ours.argmax()) == int(theirs.argmax())
        generated, token = [], int(ours.argmax())
        for _ in range(GEN_TOKENS - 1):
            generated.append(token)
            _, nxt = executor.decode(token)
            token = int(nxt[0])
        generated.append(token)
        checks.append(dict(prompt_tokens=len(row['ids']), rel_rms=rms,
                           argmax_match=argmax_match,
                           greedy_match=generated == row['greedy'],
                           ours_greedy=generated, ref_greedy=row['greedy']))
        print(f"[gate] {len(row['ids'])}tok rms {rms:.4f} argmax {argmax_match} "
              f"greedy {generated == row['greedy']}", flush=True)

    worst = max(c['rel_rms'] for c in checks)
    report = dict(model=args.model, quantization='q8_0',
                  prompts=len(checks), worst_rel_rms=worst,
                  all_argmax=all(c['argmax_match'] for c in checks),
                  all_greedy=all(c['greedy_match'] for c in checks),
                  weight_gib=model.bytes/2**30,
                  peak_alloc_gib=torch.cuda.max_memory_allocated()/2**30,
                  checks=checks)
    json.dump(report, open(args.out, 'w'), indent=1)
    print(f'[gate] worst rel-RMS {worst:.4f}; '
          f"argmax {report['all_argmax']}; greedy {report['all_greedy']}; "
          f"wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
