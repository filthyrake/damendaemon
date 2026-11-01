#!/usr/bin/env python3
"""#642 damendaemon-stack evaluation: accuracy-at-budget + TTFT/memory.

Trained-artifact path: the upstream trainer produces gate_step{N}.pt (gate
weights) + lora_step{N}.pt (backbone adapter, q/v_proj of the full-attention
layers). Serving on the damendaemon hybrid executor needs the adapter MERGED
into a copy of the pinned base snapshot (the executor has no LoRA surface);
this script performs the merge (distinct output dir, base blobs never
overwritten — the #564 lesson), then measures on OUR executor:

  1. accuracy-at-budget on NIAH-16K (the #640 protocol pool, test rows):
     full-context baseline vs gated at K in {100, 500, 1000, 2000}, greedy
     continuation, identical prompts across arms;
  2. TTFT: gated prefill wall time vs standard full prefill, same prompts,
     torch.cuda.synchronize'd — the first damendaemon-stack TokenGate number;
  3. peak CUDA memory per arm (max_memory_allocated).

The accuracy comparison is drift/accuracy per the #23 pattern, NOT
byte-exactness: pruned forwards are compared by accuracy, stated here.

Run on the 3090 host with the staged damendaemon + the trained artifacts:
  PYTHONPATH=<staged damendaemon> python token_gate_eval_642.py \
      --base /home/damen/models/Qwen3.5-0.8B-r2fc0636 \
      --gate <gate_step>.pt --lora <lora_step>.pt \
      --jsonl /home/damen/tokengate-640/datasets/niah_kamradt/niah_kamradt_all.jsonl \
      --out tokengate-642-eval.json
"""
if not __debug__:
    raise RuntimeError('run the eval under normal python, without -O: '
                       'asserts gate the merged-checkpoint contract')

import argparse
import json
import os
import shutil
import time

import torch

GEN_TOKENS = 48  # NIAH answers (~14-18 tok) follow a ~10-token echo prefix


def merge_lora(base_dir, lora_path, out_dir, alpha=32.0):
    """Merge the upstream LoRA adapter into a copy of the base snapshot.

    Adapter keys are HF-style `...q_proj.lora_A.weight` / `lora_B.weight` on
    the full-attention layers' q/v projections; delta = B @ A * alpha/r.
    Output is a full HF snapshot; the base blobs are never touched.
    """
    import safetensors.torch
    if os.path.exists(out_dir):
        raise SystemExit(f'merged dir already exists: {out_dir} (refusing to '
                         f'overwrite; pass a fresh --merged dir)')
    shutil.copytree(base_dir, out_dir)
    adapter = torch.load(lora_path, map_location='cpu', weights_only=True)
    # Upstream save format (lora.py): `...lora_A` / `...lora_B` parameters
    # (no `.weight` suffix), A=(in, rank), B=(rank, out), scaling alpha/rank,
    # applied as x @ A @ B — merged weight delta = (A @ B).T * (alpha/rank).
    # The trainer's adapter names the text backbone bare (`model.layers...`);
    # the base snapshot is multimodal-wrapped (`model.language_model.layers...`).
    def _adapter_pair(stem):
        for cand in (stem, stem.replace('model.language_model.', 'model.', 1)):
            a, b = f'{cand}.lora_A', f'{cand}.lora_B'
            if a in adapter and b in adapter:
                return adapter[a], adapter[b]
        return None
    merged = 0
    for shard in sorted(os.listdir(out_dir)):
        if not shard.endswith('.safetensors'):
            continue
        path = os.path.join(out_dir, shard)
        tensors = safetensors.torch.load_file(path)
        changed = {}
        for key in list(tensors):
            if not key.endswith('.weight'):
                continue
            pair = _adapter_pair(key[:-len('.weight')])
            if pair is not None:
                a, b = pair
                rank = a.shape[-1]
                delta = (a.float() @ b.float()).T * (alpha / rank)
                changed[key] = (tensors[key].float() + delta).to(tensors[key].dtype)
                merged += 1
        if changed:
            safetensors.torch.save_file({**tensors, **changed}, path)
    if not merged:
        raise SystemExit('no LoRA deltas matched any safetensors weight — '
                         'adapter key format changed?')
    print(f'[merge] merged {merged} projections into {out_dir}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', required=True, help='pinned base snapshot dir')
    parser.add_argument('--gate', required=True, help='gate_step*.pt')
    parser.add_argument('--lora', required=True, help='lora_step*.pt')
    parser.add_argument('--merged', default='/home/damen/models/Qwen3.5-0.8B-642-merged')
    parser.add_argument('--reuse-merged', action='store_true',
                        help='skip the LoRA merge and use the existing --merged '
                             'snapshot verbatim (verification cells)')
    parser.add_argument('--jsonl', required=True, help='NIAH pool (test rows)')
    parser.add_argument('--context', type=int, default=24576)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--ks', default='100,500,1000,2000')
    parser.add_argument('--out', default='token-gate-642-eval.json')
    args = parser.parse_args()

    if args.reuse_merged:
        if not os.path.isdir(args.merged):
            raise SystemExit(f'--reuse-merged: {args.merged} does not exist')
        merged = args.merged
    else:
        merged = merge_lora(args.base, args.lora, args.merged)

    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': '2000',
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(merged, args.context)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    assert executor.gate is not None, 'gate did not attach'
    print(f'[eval] gate attached: bias layers {executor.gate.bias_layers}')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    rows = [json.loads(line) for line in open(args.jsonl)]
    rows = [r for r in rows if r['split'] == 'test'][:args.samples]
    assert rows, 'no test rows'
    # Protocol geometry (#640/#642): article = haystack body (prunable);
    # the trailing "\n\nQuestion: ..." is a non-prunable suffix.
    prompts = []
    for row in rows:
        body_ids = tokenizer(row['body'], add_special_tokens=False)['input_ids']
        suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                           add_special_tokens=False)['input_ids']
        prompts.append((body_ids + suffix, len(body_ids), row['answer']))
    print(f'[eval] {len(prompts)} test prompts, '
          f'{max(len(p[0]) for p in prompts)} max tokens')

    record = {'model': 'Qwen3.5-0.8B pinned r2fc0636 + #642 bounded gate',
              'gate': os.path.basename(args.gate), 'lora': os.path.basename(args.lora),
              'samples': len(prompts), 'gen_tokens': GEN_TOKENS, 'arms': {},
              'convention': 'accuracy/drift comparison per the #23 pattern; '
                            'pruned forwards are compared by accuracy, never '
                            'byte-exactness'}

    slots = torch.tensor([0], device=executor.device)

    def greedy(prefill_fn, after_prefill=None):
        """(ttft_s, generated_ids) per prompt on a freshly-reset slot.

        Each prompt resets slot 0's length AND gate state (a stale kept-set
        from the previous prompt would leak into the decode hole mask). The
        full-context arm marks the whole prompt valid post-prefill — the
        #641 ungated semantics, which makes decode exactly the base model.
        """
        out = []
        for ids, article_end, _ in prompts:
            executor.lengths[0] = 0
            executor._gate_reset_slot(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = prefill_fn(ids, article_end)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - start
            if after_prefill is not None:
                after_prefill(ids)
            generated, token = [], int(logits.argmax(-1)[0])
            for _ in range(GEN_TOKENS):
                generated.append(token)
                step, nxt = executor.decode(token)
                token = int(nxt[0])
            out.append((ttft, generated))
        return out

    def score(arm_name, results):
        correct = 0
        texts = []
        for (_, generated), (_, _, answer) in zip(results, prompts):
            text = tokenizer.decode(generated)
            texts.append(text)
            correct += answer.lower() in text.lower()
        times = [t for t, _ in results]
        record['arms'][arm_name] = {
            'acc': correct / len(prompts),
            'correct': correct,
            'ttft_s_mean': sum(times) / len(times),
            'ttft_s_min': min(times), 'ttft_s_max': max(times),
            'peak_alloc_gib': torch.cuda.max_memory_allocated() / 2**30,
            'first_tokens': [g[0] for _, g in results],
            'samples': texts,
        }
        print(f"[eval] {arm_name}: acc={correct}/{len(prompts)} "
              f"ttft_mean={sum(times)/len(times):.3f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    # gated arms (K sweep)
    for k in [int(x) for x in args.ks.split(',')]:
        torch.cuda.reset_peak_memory_stats()
        results = greedy(lambda ids, end, k=k: executor.gated_prefill(ids, 0, 0, end, k))
        score(f'gated_k{k}', results)

    # full-context baseline through the standard path (gate state fully
    # ungated — decode is exactly the base model)
    torch.cuda.reset_peak_memory_stats()

    def full_prefill(ids, _end):
        tensor = torch.tensor([ids], device=executor.device)
        positions = torch.arange(len(ids), device=executor.device)[None]
        logits = executor.forward(tensor, positions, slots, True)
        # forward() is the low-level path: it does not do the slot-length
        # bookkeeping gated_prefill performs internally — decode_batch would
        # reject the slot as empty. Set it explicitly (drift-script pattern).
        executor.lengths[0] = len(ids)
        return logits

    def mark_ungated(ids):
        executor._gate_mark_ungated(0, len(ids))

    results = greedy(full_prefill, after_prefill=mark_ungated)
    score('full_context', results)

    with open(args.out, 'w') as handle:
        json.dump(record, handle, indent=1)
    print(f'[eval] wrote {args.out}')


if __name__ == '__main__':
    main()
