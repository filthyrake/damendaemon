#!/usr/bin/env python3
"""#642 bisection: our gated executor vs upstream's harness, same rows,
same checkpoint+gate, one process — first tokens (prefill argmax), kept
sets (original positions), short greedy texts.

Axes compared per row:
  1. kept sets       — selection parity on THIS checkpoint (0.8B step-1200,
                       l1d placement; #641 verified selection only on the 2B
                       step-18000 checkpoint);
  2. first token     — prefill phase-2 agreement (argmax + top-10 from ours);
  3. short text      — decode continuation agreement.

If kept sets match but first tokens diverge, the bug is phase-2 attention
prefill; if first tokens match but texts diverge, decode; if kept sets
differ, selection/aggregate on this checkpoint.
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
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--out', default='/tmp/g642r/bisect.json')
    args = parser.parse_args()

    # ── our executor (this tree) ───────────────────────────────────────
    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.k),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(args.merged, 24576)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    rows = [json.loads(line) for line in open(args.jsonl)
            if json.loads(line)['split'] == 'test'][:args.rows]
    prompts = []
    for row in rows:
        body = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        prompts.append((body + suffix, len(body), row['answer']))

    ours = []
    for ids, end, answer in prompts:
        executor.lengths[0] = 0
        executor._gate_reset_slot(0)
        logits = executor.gated_prefill(ids, 0, 0, end, args.k)
        topv, topi = torch.softmax(logits[0].float(), -1).topk(10)
        kept = torch.nonzero(executor.gate_valid[3][0]).flatten().tolist()
        tokens, token = [], int(logits.argmax(-1)[0])
        for _ in range(args.steps):
            tokens.append(token)
            _, nxt = executor.decode(token)
            token = int(nxt[0])
        tokens.append(token)
        ours.append(dict(n_prompt=len(ids), first=tokens[0], tokens=tokens,
                         text=tokenizer.decode(tokens),
                         top10=[(int(i), float(v)) for i, v in zip(topi, topv)],
                         kept_n=len(kept), kept=kept))
    print('[ours] done', len(ours), 'rows')

    # ── upstream harness, same process, same card ──────────────────────
    sys.path.insert(0, args.upstream)
    from transformers import AutoModelForCausalLM
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

    theirs = []
    for ids, end, answer in prompts:
        res = generate_batch(umodel, gate, tokenizer, [ids], [0], [end], 1,
                             args.k, args.steps, decode_phase2_form=True,
                             prune_below_gate=True)
        kept_up = [p for p in res.kept_pos[0].tolist() if p >= 0]
        theirs.append(dict(text=res.answers[0], n_gen=res.n_gen[0],
                           kept_n=len(kept_up), kept=sorted(kept_up)))
    print('[upstream] done', len(theirs), 'rows')

    report = []
    for (ids, end, answer), o, t in zip(prompts, ours, theirs):
        kept_equal = o['kept'] == t['kept']
        report.append(dict(answer=answer, kept_equal=kept_equal,
                           kept_ours=o['kept_n'], kept_up=t['kept_n'],
                           kept_only_ours=o['kept_n'] - len(set(o['kept']) & set(t['kept'])),
                           first_ours=o['first'], text_ours=o['text'],
                           text_up=t['text'], n_gen_up=t['n_gen'],
                           prefix_match=o['text'].strip()[:12] == t['text'].strip()[:12],
                           top10=o['top10']))
    summary = dict(rows=len(report),
                   kept_all_equal=all(r['kept_equal'] for r in report),
                   prefix_any=sum(r['prefix_match'] for r in report))
    print(json.dumps(summary))
    json.dump(dict(summary=summary, rows=report, ours=ours, theirs=theirs),
              open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
