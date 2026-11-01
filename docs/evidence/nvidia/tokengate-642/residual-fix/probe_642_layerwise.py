#!/usr/bin/env python3
"""#642 layerwise bisection: our gated phase-1 boundary and phase-2 layer
outputs vs upstream's, on ONE row with upstream's exact kept set forced.

For row 2 (a row where upstream's harness produces a real answer at
K=2000), capture:
  - upstream: hooked per-layer outputs of prefill_batch_for_decode's phase-2
    loop (layers gate+1..end), plus the pruned boundary (phase-1 output at
    kept positions, after the gate layer's MLP half);
  - ours: the same quantities from our executor with gate.select patched to
    upstream's kept set;
and report relative RMS per layer. The first layer with a large divergence
localizes the integration defect (boundary vs linear layer vs gated
attention vs head).
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import os
import sys

import torch


def relrms(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', default='/home/damen/tokengate-640')
    parser.add_argument('--base', default='/home/damen/models/Qwen3.5-0.8B-r2fc0636')
    parser.add_argument('--merged', default='/home/damen/models/Qwen3.5-0.8B-642-merged')
    parser.add_argument('--gate', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--bisect', default='/tmp/g642r/bisect.json')
    parser.add_argument('--row', type=int, default=2)
    parser.add_argument('--k', type=int, default=2000)
    parser.add_argument('--out', default='/tmp/g642r/layerwise.json')
    args = parser.parse_args()

    bisect = json.load(open(args.bisect))
    args.row = args.row  # kept set comes from the fresh upstream call below

    # ── upstream pass with hooks ───────────────────────────────────────
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
    from tokengate.e2e_gate.gate_module.gate_forward import (
        prefill_batch_for_decode, split_gate_layer_mlp,
        gate_branch_hs, _ssmax_ratio)

    rows = [json.loads(line) for line in open(args.jsonl)
            if json.loads(line)['split'] == 'test']
    row = rows[args.row]
    body = tokenizer(row['body'], add_special_tokens=False)['input_ids']
    suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                       add_special_tokens=False)['input_ids']
    ids = body + suffix

    captures = {}

    def hook(name):
        def fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            entered = inputs[0] if inputs else None
            captures.setdefault(name, []).append((entered.detach(), hidden.detach()))
        return fn

    n_layers = len(umodel.model.layers)
    handles = [umodel.model.layers[i].register_forward_hook(hook(f'up_layer{i}'))
               for i in range(1, n_layers)]
    first_toks, cache, kept_pos, n_kepts, kclamp = prefill_batch_for_decode(
        umodel, gate, [ids], [0], [len(body)], 1, args.k, prune_below_gate=True)
    for h in handles:
        h.remove()
    if 'up_layer2' not in captures:
        raise SystemExit(f'phase-2 hooks did not fire; captured keys: '
                         f'{sorted(captures)}')
    # upstream boundary: the input to phase-2 layer 2 on its single call —
    # the gathered kept-position hidden states after the gate layer's MLP
    # half (whichever side of the split applied it).  The gate layer itself
    # runs via attn_only_forward and does not fire module forward hooks, so
    # layer 2's input is the reliable capture point.
    # The kept set is taken from THIS run's prefill (upstream's own selection
    # is not bit-stable across runs at K=2000 — the bisect run kept 2013,
    # this run can keep a slightly different set).
    upstream_kept = [p for p in kept_pos[0].tolist() if p >= 0][:n_kepts[0]]
    n_kept = n_kepts[0]
    print(f'upstream kept {n_kept} positions this run '
          f'(bisect had {bisect["theirs"][args.row]["kept_n"]})')
    up_boundary = captures['up_layer2'][0][0][None][:, :n_kept]   # (1, n_kept, d)
    split = split_gate_layer_mlp(gate)
    print('split gate MLP:', bool(split))
    up_first = int(first_toks[0])
    up_layers = {i: captures[f'up_layer{i}'][-1][1][0][:n_kept] for i in range(2, n_layers)}
    print('upstream first token:', up_first, repr(tokenizer.decode([up_first])))

    del umodel, captures
    torch.cuda.empty_cache()

    # ── our executor, forced kept set, layerwise ────────────────────────
    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.k),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    from damendaemon.token_gate import TokenGate
    model = load_model(args.merged, 24576)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

    end = len(body)
    forced = torch.tensor([p for p in upstream_kept if 0 <= p < end],
                          dtype=torch.long, device=executor.device)
    # gated_prefill's kept set: the forced ARTICLE positions plus every
    # non-article position (prefix + question suffix), sorted.
    all_positions = torch.arange(len(ids), device=executor.device)
    article_mask = torch.zeros(len(ids), dtype=torch.bool, device=executor.device)
    article_mask[0:end] = True
    kept_positions = torch.sort(torch.cat((all_positions[~article_mask], forced))).values
    assert kept_positions.numel() == n_kept, (kept_positions.numel(), n_kept)

    slots = torch.tensor([0], device=executor.device)
    ids_t = torch.tensor([ids], device=executor.device)
    positions = torch.arange(len(ids), device=executor.device)[None]
    boundary_full = executor.forward_boundary(
        executor.embedding(ids_t), positions, slots, True)
    scores = executor.gate.score_tokens(
        boundary_full, executor.w['1.norm2'], executor.eps)
    article = torch.zeros(len(ids), dtype=torch.bool, device=executor.device)
    article[0:end] = True
    bias_all = executor.gate.bias(scores, article)          # (1, n, rows, heads)
    x = boundary_full.index_select(1, kept_positions)
    x = executor._mlp_half(x, 1)
    ours_boundary = x
    ratio = TokenGate.kept_ratio(kept_positions)
    bias = bias_all[0][kept_positions]                      # (kept, rows, heads)
    layer_caps = {}
    for i in range(2, executor.layer_end):
        if executor.c['layer_types'][i] == 'linear_attention':
            h = executor.norm(x, executor.w[f'{i}.norm1'])
            x = x + executor.linear_attention(h, i, slots, True, 0)
        else:
            r = executor.gate.bias_layers.index(i)
            h = executor.norm(x, executor.w[f'{i}.norm1'])
            x = x + executor._gated_attention(h, i, kept_positions, slots,
                                              ratio, bias[:, r])
        layer_caps[i] = x.detach()
    logits = executor.output_logits(x[:, -1])
    ours_first = int(logits.argmax(-1)[0])
    print('ours-forced first token:', ours_first, repr(tokenizer.decode([ours_first])))

    # ── compare ────────────────────────────────────────────────────────
    report = {'boundary_rms': relrms(ours_boundary, up_boundary),
              'boundary_up_norm': float(up_boundary.float().norm()),
              'ours_first': ours_first, 'up_first': up_first,
              'split': bool(split), 'kept_n': len(upstream_kept),
              'layers': []}
    for i in sorted(set(up_layers) & set(layer_caps)):
        rms = relrms(layer_caps[i], up_layers[i])
        report['layers'].append(dict(layer=i,
                                     type=executor.c['layer_types'][i],
                                     rms=rms,
                                     up_norm=float(up_layers[i].float().norm()),
                                     ours_norm=float(layer_caps[i].float().norm())))
        print(f'layer {i:2d} {executor.c["layer_types"][i]:16s} '
              f'rel-rms {rms:8.4f}  up_norm {float(up_layers[i].float().norm()):10.1f}')
    print('boundary rel-rms:', report['boundary_rms'])
    json.dump(report, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
