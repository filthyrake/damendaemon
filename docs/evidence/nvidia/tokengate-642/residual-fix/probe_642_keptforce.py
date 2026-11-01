#!/usr/bin/env python3
"""#642 kept-set forcing: run OUR executor with UPSTREAM's exact kept set.

The bisect cell showed kept sets differ by ~10/2013 positions per row (same
counts, near-threshold flips) while component selection parity was proven
on the 2B checkpoint (#641). This cell separates selection sensitivity from
integration error:

  - upstream generate_batch (48 tokens) per row  -> their kept set + text;
  - our gated_prefill with `gate.select` patched to return upstream's kept
    ARTICLE positions verbatim, then 48 greedy tokens.

If our texts/accuracy converge with upstream's, the phase-2/decode
integration is correct and the 0/24-vs-5/24 residual is selection
sensitivity to backbone numerics at this weak checkpoint. If they still
diverge, phase-2/decode has a real defect.
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', default='/home/damen/tokengate-640')
    parser.add_argument('--base', default='/home/damen/models/Qwen3.5-0.8B-r2fc0636')
    parser.add_argument('--merged', default='/home/damen/models/Qwen3.5-0.8B-642-merged')
    parser.add_argument('--gate', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--k', type=int, default=2000)
    parser.add_argument('--rows', type=int, default=12)
    parser.add_argument('--steps', type=int, default=48)
    parser.add_argument('--out', default='/tmp/g642r/keptforce.json')
    args = parser.parse_args()

    # ── upstream harness first (their kept sets + answers) ─────────────
    sys.path.insert(0, args.upstream)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    umodel = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map='cuda')
    from tokengate.e2e_gate.gate_module.lora import (apply_lora_to_layers,
                                                     load_lora_state_dict)
    apply_lora_to_layers(umodel, [3, 7, 11, 15, 19, 23], 16, 32.0,
                         targets=['q_proj', 'v_proj'])
    load_lora_state_dict(umodel, torch.load(args.lora, map_location='cpu',
                                            weights_only=True))
    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        patch_model_with_flex_attention)
    umodel = patch_model_with_flex_attention(umodel)
    from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
    gate, _ = load_gate(args.gate)
    gate = gate.to('cuda')
    gate.gate_branch.to('cuda')
    from tokengate.e2e_gate.decode.gate_decode import generate_batch

    rows = [json.loads(line) for line in open(args.jsonl)
            if json.loads(line)['split'] == 'test'][:args.rows]
    prompts = []
    for row in rows:
        body = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        prompts.append((body + suffix, len(body), row['answer']))

    theirs = []
    for ids, end, answer in prompts:
        res = generate_batch(umodel, gate, tokenizer, [ids], [0], [end], 1,
                             args.k, args.steps, decode_phase2_form=True,
                             prune_below_gate=True)
        theirs.append(dict(text=res.answers[0], n_gen=res.n_gen[0],
                           kept=sorted(p for p in res.kept_pos[0].tolist() if p >= 0)))
    up_correct = sum(answer.lower() in t['text'].lower()
                     for (_, _, answer), t in zip(prompts, theirs))
    print(f'[upstream] {up_correct}/{len(prompts)} correct, '
          f'{sum(not t["text"] for t in theirs)} empty')

    del umodel
    torch.cuda.empty_cache()

    # ── our executor with forced kept sets ─────────────────────────────
    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.k),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.merged, 24576)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

    ours = []
    original_select = executor.gate.select
    for (ids, end, answer), t in zip(prompts, theirs):
        upstream_kept = t['kept']
        forced = torch.tensor([p for p in upstream_kept if 0 <= p < end],
                              dtype=torch.long, device=executor.device)

        def select(scores, article_mask, keep, forced=forced):
            return forced.sort().values

        executor.gate.select = select
        executor.lengths[0] = 0
        executor._gate_reset_slot(0)
        logits = executor.gated_prefill(ids, 0, 0, end, args.k)
        tokens, token = [], int(logits.argmax(-1)[0])
        for _ in range(args.steps):
            tokens.append(token)
            _, nxt = executor.decode(token)
            token = int(nxt[0])
        tokens.append(token)
        ours.append(dict(tokens=tokens, text=tokenizer.decode(tokens),
                         kept_used=len(forced)))
    executor.gate.select = original_select

    forced_correct = sum(answer.lower() in o['text'].lower()
                         for (_, _, answer), o in zip(prompts, ours))
    exact = sum(o['text'].strip() == t['text'].strip()
                for o, t in zip(ours, theirs))
    print(f'[ours-forced] {forced_correct}/{len(prompts)} correct; '
          f'{exact}/{len(prompts)} texts exactly match upstream')
    report = []
    for (ids, end, answer), o, t in zip(prompts, ours, theirs):
        report.append(dict(answer=answer, upstream_correct=answer.lower() in t['text'].lower(),
                           forced_correct=answer.lower() in o['text'].lower(),
                           text_match=o['text'].strip() == t['text'].strip(),
                           ours=o['text'], upstream=t['text'], n_gen_up=t['n_gen']))
    summary = dict(rows=len(report), upstream_correct=up_correct,
                   forced_correct=forced_correct, exact_text=exact)
    print(json.dumps(summary))
    json.dump(dict(summary=summary, rows=report), open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
