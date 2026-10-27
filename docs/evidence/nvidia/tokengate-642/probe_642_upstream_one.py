#!/usr/bin/env python3
"""#642 diagnostic B: the SAME prompt+checkpoint through upstream's harness.

Upstream path (the one whose step-1200 in-training eval scored 0.48-0.59 at
K=2000 on this exact checkpoint): live-LoRA model + load_gate +
gate_decode.generate_batch with prune_below_gate=True (our #642 patch) and
decode_phase2_form=True. A/B against our executor's gated path on the same
row: if upstream is sane here, our #641 gated integration diverges; if
upstream matches our counting garbage, the checkpoint itself is the finding.
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
    parser.add_argument('--gate', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--k', type=int, default=2000)
    args = parser.parse_args()

    sys.path.insert(0, args.upstream)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    # live LoRA, exactly as the trainer's eval did
    from tokengate.e2e_gate.gate_module.lora import (apply_lora_to_layers,
                                                     load_lora_state_dict)
    apply_lora_to_layers(model, [3, 7, 11, 15, 19, 23], 16, 32.0,
                         targets=['q_proj', 'v_proj'])
    load_lora_state_dict(model, torch.load(args.lora, map_location='cpu',
                                           weights_only=True))
    # the trainer's mandatory prep (train_e2e_gate.py:3775): sets per-layer
    # type attributes from config.layer_types — without it _apply_gate_bias
    # sees every layer as full-attention (the #640 layer_type dispatch bug).
    from tokengate.e2e_gate.attn.qwen3_flex_attention import (
        patch_model_with_flex_attention)
    model = patch_model_with_flex_attention(model)

    from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
    gate, metadata = load_gate(args.gate)
    gate = gate.to('cuda')
    gate.gate_branch.to('cuda')

    from tokengate.e2e_gate.decode.gate_decode import generate_batch
    rows = [json.loads(l) for l in open(args.jsonl)]
    rows = [r for r in rows if r['split'] == 'test'][:24]
    correct, empties = 0, 0
    for row in rows:
        body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        ids = body_ids + suffix
        res = generate_batch(model, gate, tokenizer, [ids], [0], [len(body_ids)],
                             1, args.k, 48, decode_phase2_form=True,
                             prune_below_gate=True)
        text = res.answers[0]
        if not text:
            empties += 1
        correct += row['answer'].lower() in text.lower()
    print(f'upstream K={args.k}: {correct}/{len(rows)} correct, '
          f'{empties} empty generations')
    for row in rows[:2]:
        body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        ids = body_ids + suffix
        res = generate_batch(model, gate, tokenizer, [ids], [0], [len(body_ids)],
                             1, args.k, 48, decode_phase2_form=True,
                             prune_below_gate=True)
        print('example:', repr(res.answers[0][:200]))


if __name__ == '__main__':
    main()
