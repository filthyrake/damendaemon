#!/usr/bin/env python3
"""#642 diagnostic C: full-context on the LIVE-LoRA adapted model (no merge).

The merge-fidelity control: plain model.generate on base + live adapter.
Compare against our executor's full-context on the MERGED snapshot (10/24)
and on the base (22/24). If live-LoRA ~= merged, the merge is faithful and
the full-context drop is genuine co-adaptation; if live-LoRA ~= base, the
merge damaged the backbone.
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', default='/home/damen/tokengate-640')
    parser.add_argument('--base', default='/home/damen/models/Qwen3.5-0.8B-r2fc0636')
    parser.add_argument('--lora', required=True)
    parser.add_argument('--jsonl', required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.upstream)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    from tokengate.e2e_gate.gate_module.lora import (apply_lora_to_layers,
                                                     load_lora_state_dict)
    apply_lora_to_layers(model, [3, 7, 11, 15, 19, 23], 16, 32.0,
                         targets=['q_proj', 'v_proj'])
    load_lora_state_dict(model, torch.load(args.lora, map_location='cpu',
                                           weights_only=True))

    rows = [json.loads(l) for l in open(args.jsonl)]
    rows = [r for r in rows if r['split'] == 'test'][:24]
    correct = 0
    for row in rows:
        body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                          add_special_tokens=False)['input_ids']
        ids = body_ids + suffix
        out = model.generate(torch.tensor([ids], device='cuda'),
                             max_new_tokens=48, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][len(ids):])
        correct += row['answer'].lower() in text.lower()
    print(f'live-LoRA full-context: {correct}/{len(rows)} correct')


if __name__ == '__main__':
    main()
