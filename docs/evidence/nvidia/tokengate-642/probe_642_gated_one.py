#!/usr/bin/env python3
"""#642 diagnostic: one gated prompt through our executor — kept set + text.

Localizes the gated-path divergence: layer-1 boundary and selection are
bit-verified (#641) and LoRA touches only layers 3,7,...,23, so the kept set
must match upstream's. Prints whether the needle token range survives, and
what the model actually generates at K=2000.
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/damen/models/Qwen3.5-0.8B-642-merged')
    parser.add_argument('--gate', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--context', type=int, default=24576)
    parser.add_argument('--k', type=int, default=2000)
    args = parser.parse_args()

    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.k),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.model, args.context)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/damen/models/Qwen3.5-0.8B-r2fc0636')
    row = next(json.loads(l) for l in open(args.jsonl) if json.loads(l)['split'] == 'test')
    body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
    suffix = tokenizer(f"\n\nQuestion: {row['question']}", add_special_tokens=False)['input_ids']
    ids = body_ids + suffix
    answer = row['answer']

    # needle = the answer sentence inside the body ("The best thing to do in X is <answer>.")
    needle_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    print(f'prompt {len(ids)} tokens, article [0, {len(body_ids)}), K={args.k}')
    print(f'answer: {answer!r}')

    logits = executor.gated_prefill(ids, 0, 0, len(body_ids), args.k)
    kept = torch.nonzero(executor.gate_valid[3][0]).flatten()
    print(f'kept {kept.numel()} positions')

    # where does the needle live in the body?
    body_t = torch.tensor(body_ids)
    needle_t = torch.tensor(needle_ids)
    # find first occurrence by sliding compare
    found = None
    for i in range(len(body_ids) - len(needle_ids)):
        if torch.equal(body_t[i:i + len(needle_ids)], needle_t):
            found = i
            break
    if found is None:
        print('needle token sequence NOT found verbatim (tokenizer split difference)')
    else:
        inside = any(found <= p < found + len(needle_ids) for p in kept.tolist())
        # nearest kept position to the needle
        near = min(kept.tolist(), key=lambda p: abs(p - found))
        print(f'needle at [{found}, {found + len(needle_ids)}); needle kept: {inside}; '
              f'nearest kept pos {near} (dist {abs(near - found)})')

    generated, token = [], int(logits.argmax(-1)[0])
    for _ in range(48):
        generated.append(token)
        _, nxt = executor.decode(token)
        token = int(nxt[0])
    print('generated:', repr(tokenizer.decode(generated)))
    print('answer in text:', answer.lower() in tokenizer.decode(generated).lower())


if __name__ == '__main__':
    main()
