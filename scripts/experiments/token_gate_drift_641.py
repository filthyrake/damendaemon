#!/usr/bin/env python3
"""#641 numerical reference record: our TokenGate vs the upstream E2EGate.

Pinned hybrid checkpoint: Qwen/Qwen3.5-2B + the released gate
liana313/tokengate-niah-qwen3.5-2b (gate_step18000.pt — gate at DeltaNet
layer 1, the exact paper checkpoint, #640 pins).

Scope, stated honestly: upstream's own hybrid model-forward is broken as
shipped (#640: three confirmed bugs in their gated hybrid inference path),
so a model-forward parity is not producible from upstream main. This record
is therefore COMPONENT-level on real activations: our executor's real
layer-1 phase boundary (the hybrid prefill path, eager torch kernels)
feeds BOTH gate implementations, and every gate component is compared
against the upstream implementation loaded from the same checkpoint file:

  1. LN2 boundary normalization  (our norm() vs Qwen3_5RMSNorm)
  2. scoring branch              (our branch vs upstream gate_branch)
  3. E2EGate bias                (our bias() vs upstream E2EGate.forward)
  4. top-K selection             (our select() vs upstream aggregate+topk)
  5. kept-ratio temperature      (our kept_ratio vs upstream _ssmax_ratio)

Drift is relative-RMS per the #23/#535 convention; selection is compared
exactly (index sets). The phase-2 attention form itself (temperature-scaled
logits + per-key bias) has no standalone upstream entry point on the hybrid
(it lives inside their broken decode path) and is covered by the CPU
property suite instead — recorded as a limitation, not a claim.

Run on the 3090 host (guarded, power-capped):
  PYTHONPATH=<staged damendaemon> python token_gate_drift_641.py \
      --upstream /home/damen/tokengate-640 --out drift.json
"""
import argparse
import json
import math
import os
import sys

import torch


def rel_rms(reference, actual):
    """Relative-RMS drift per the #23/#535 convention."""
    reference, actual = reference.float(), actual.float()
    scale = reference.square().mean().sqrt().clamp_min(1e-12)
    return float((reference - actual).square().mean().sqrt() / scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--upstream', required=True,
                        help='upstream TokenGate clone root (sys.path)')
    parser.add_argument('--model', default='Qwen/Qwen3.5-2B')
    parser.add_argument('--gate-hf', default='liana313/tokengate-niah-qwen3.5-2b')
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--article-tokens', type=int, default=2048)
    parser.add_argument('--keep', type=int, default=128)
    parser.add_argument('--out', default='token-gate-drift.json')
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.upstream))

    from huggingface_hub import snapshot_download

    # ── upstream gate, loaded through upstream's own loader ──────────────────
    from tokengate.e2e_gate.gate_module.e2e_gate_module import load_gate
    gate_dir = snapshot_download(args.gate_hf)
    gate_path = next(os.path.join(gate_dir, name) for name in sorted(os.listdir(gate_dir))
                     if name.startswith('gate_step') and name.endswith('.pt'))
    upstream_gate, metadata = load_gate(gate_path)
    gate_layer = int(metadata['gate_layer'])
    assert upstream_gate.gate_branch is not None, 'upstream loader attached no branch'
    print(f'[drift] upstream gate: layer={gate_layer} '
          f'heads={upstream_gate.n_heads} rows={upstream_gate.n_layers_bias}')

    # ── our executor on the same base model, gate from the same file ────────
    model_dir = snapshot_download(args.model)
    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': str(gate_layer),
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.keep),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': gate_path})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    model = load_model(model_dir, args.context)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')
    gate = executor.gate
    upstream_gate = upstream_gate.to(executor.device)
    # the loader attaches gate_branch outside the module registry; move it
    # explicitly or the MLP weights stay on CPU.
    upstream_gate.gate_branch.to(executor.device)
    print(f'[drift] our gate: bias layers={gate.bias_layers} keep={gate.keep}')

    # ── a real prompt: an NIAH haystack body as the prunable article ────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = os.path.join(args.upstream, 'examples', 'data', 'niah_32k_samples.jsonl')
    body = json.loads(open(samples).readline())['body']
    article_ids = tokenizer(body, add_special_tokens=False)['input_ids'][:args.article_tokens]
    prefix = tokenizer('Answer the question about the document.\n\n',
                       add_special_tokens=False)['input_ids']
    suffix = tokenizer('\n\nQuestion: What is the best thing to do?',
                       add_special_tokens=False)['input_ids']
    prompt = prefix + article_ids + suffix
    article_start, article_end = len(prefix), len(prefix) + len(article_ids)
    print(f'[drift] prompt: {len(prompt)} tokens, article span '
          f'[{article_start}, {article_end})')

    # ── our real phase boundary (hybrid prefill path, eager) ────────────────
    ids = torch.tensor([prompt], device=executor.device)
    positions = torch.arange(len(prompt), device=executor.device)[None]
    slots = torch.tensor([0], device=executor.device)
    boundary = executor.forward_boundary(executor.embedding(ids), positions, slots, True)
    record = {'model': args.model, 'gate_checkpoint': os.path.basename(gate_path),
              'gate_layer': gate_layer, 'prompt_tokens': len(prompt),
              'article_span': [article_start, article_end], 'keep': args.keep,
              'upstream_commit': '6e51aac', 'components': {}}

    # 1. LN2 normalization: ours vs the real Qwen3.5 RMSNorm. Qwen3.5 uses
    #    the Gemma-style (1 + w) offset norm: HF checkpoints store w and
    #    multiply by (1 + w), which is exactly the executor's take()+1
    #    storage with a plain multiply. Reference feeds the raw checkpoint
    #    value into Qwen3_5RMSNorm.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    raw = executor.w[f'{gate_layer}.norm2'] - 1  # executor stores 1 + w
    upstream_norm = Qwen3_5RMSNorm(raw.shape[-1], eps=executor.eps).to(boundary.device)
    with torch.no_grad():
        upstream_norm.weight.copy_(raw.float())
        theirs_z = upstream_norm(boundary.to(torch.float32))
    ours_z = executor.norm(boundary, executor.w[f'{gate_layer}.norm2'], torch.float32)
    record['components']['ln2_boundary'] = rel_rms(theirs_z, ours_z)

    # 2. Scoring branch: same z into both implementations.
    with torch.no_grad():
        theirs_scores = upstream_gate.gate_branch(ours_z.to(torch.float32))
    ours_scores = gate._branch(ours_z)
    record['components']['branch_scores'] = rel_rms(theirs_scores.squeeze(-1), ours_scores)

    # 3. E2EGate bias: upstream forward over the article rows, ours on the
    #    full sequence with the article mask (same normalization axis).
    scores_full = gate.score_tokens(boundary, executor.w[f'{gate_layer}.norm2'], executor.eps)
    article = torch.zeros(len(prompt), dtype=torch.bool, device=executor.device)
    article[article_start:article_end] = True
    article_scores = scores_full[0][article]
    with torch.no_grad():
        theirs_bias = upstream_gate(article_scores.unsqueeze(-1).to(torch.float32))
    ours_bias = gate.bias(scores_full, article)[0][article]
    record['components']['gate_bias'] = rel_rms(theirs_bias.reshape(-1), ours_bias.reshape(-1))

    # 4. Selection: exact index-set comparison (upstream aggregate + topk).
    with torch.no_grad():
        upstream_agg = upstream_gate.aggregate(theirs_bias)
        upstream_top = torch.topk(upstream_agg.flatten(), min(args.keep,
                              article_scores.numel())).indices.sort().values
        upstream_kept = torch.nonzero(article).flatten()[upstream_top]
    ours_kept = gate.select(scores_full, article, args.keep)
    record['components']['selection_exact_match'] = bool(
        torch.equal(upstream_kept.cpu(), ours_kept.cpu()))
    record['components']['selection_n'] = int(ours_kept.numel())

    # 5. Kept-ratio temperature: upstream _ssmax_ratio on the same kept set.
    from tokengate.e2e_gate.gate_module.gate_forward import _ssmax_ratio
    theirs_ratio = _ssmax_ratio(ours_kept[None])
    ours_ratio = gate.kept_ratio(ours_kept)[None]
    record['components']['kept_ratio'] = rel_rms(theirs_ratio, ours_ratio)

    # ── mechanism evidence on the real hybrid: one gated prefill end to end
    #    (no upstream comparison — their model forward is blocked, #640). ──
    logits = executor.gated_prefill(prompt, 0, article_start, article_end, args.keep)
    kept_count = int(executor.gate_valid[gate.bias_layers[-1]][0].sum())
    record['mechanism'] = {
        'kept_tokens': kept_count,
        'prompt_tokens': len(prompt),
        'logits_finite': bool(torch.isfinite(logits).all()),
        'first_token': int(logits.argmax(-1)[0]),
    }
    decode_token = int(logits.argmax(-1)[0])
    finite = True
    for _ in range(8):
        step, nxt = executor.decode(decode_token)
        finite = finite and bool(torch.isfinite(step).all())
        decode_token = int(nxt[0])
    record['mechanism']['decode_steps_finite'] = finite

    with open(args.out, 'w') as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record, indent=1))
    print(f'[drift] wrote {args.out}')


if __name__ == '__main__':
    main()
