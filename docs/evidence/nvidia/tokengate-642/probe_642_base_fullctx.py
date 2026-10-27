#!/usr/bin/env python3
"""#642 diagnostic: full-context arm on the UNMERGED base through our executor.

Separates two hypotheses for the eval discrepancy (gated 0/24, merged
full-context 10/24 vs upstream-harness 0.48-0.59 gated / ~0.9 full):
  A. our eval harness (scoring/greedy/decode) under-scores -> base scores low too
  B. the LoRA merge damages the backbone -> base scores ~0.9, merged doesn't
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import time

import torch

GEN_TOKENS = 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--context', type=int, default=24576)
    args = parser.parse_args()

    os_env = {'DAMENDAEMON_DEVICE': 'cuda'}
    import os
    for k, v in os_env.items():
        os.environ[k] = v
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.model, args.context)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.jsonl)]
    rows = [r for r in rows if r['split'] == 'test'][:args.samples]
    prompts = []
    for row in rows:
        body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        prompts.append((body_ids + suffix, row['answer']))

    slots = torch.tensor([0], device=executor.device)
    correct, examples = 0, []
    for ids, answer in prompts:
        executor.lengths[0] = 0
        tensor = torch.tensor([ids], device=executor.device)
        positions = torch.arange(len(ids), device=executor.device)[None]
        logits = executor.forward(tensor, positions, slots, True)
        executor.lengths[0] = len(ids)
        generated, token = [], int(logits.argmax(-1)[0])
        for _ in range(GEN_TOKENS):
            generated.append(token)
            _, nxt = executor.decode(token)
            token = int(nxt[0])
        text = tokenizer.decode(generated)
        ok = answer.lower() in text.lower()
        correct += ok
        if len(examples) < 3:
            examples.append({'answer': answer, 'generated': text, 'ok': ok})
    print(json.dumps({'model': args.model, 'acc': correct / len(prompts),
                      'correct': correct, 'n': len(prompts), 'examples': examples},
                     indent=1))


if __name__ == '__main__':
    main()
