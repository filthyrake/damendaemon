#!/usr/bin/env python3
"""#642 cross-call isolation for the layer-2 divergence.

The layerwise cell showed: phase-2 entry (boundary) agrees at 0.7% overall,
yet layer 2 (the first linear-attention layer over the kept sequence)
already diverges to 68% rel-RMS. This cell separates the two explanations:

  A. run UPSTREAM's layer 2 on OUR boundary      -> if it matches upstream's
     own layer-2 output, our boundary is locally fine and OUR layer-2 math
     differs; if it instead matches our output, the boundary difference is
     concentrated where it matters;
  B. run OUR linear_attention on UPSTREAM's boundary -> symmetric evidence;
  C. per-position boundary rel-RMS profile (article rows vs suffix rows vs
     head/tail buckets) to see where the 0.7% lives;
  D. dumps both boundaries and layer-2 outputs to .pt for offline analysis.
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
    parser.add_argument('--k', type=int, default=2000)
    parser.add_argument('--row', type=int, default=2)
    parser.add_argument('--out', default='/tmp/g642r/crosslayer.json')
    parser.add_argument('--dump', default='/tmp/g642r/crosslayer.pt')
    args = parser.parse_args()

    # ── upstream pass with hooks (layers 2..23), keep the model alive ──
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
        prefill_batch_for_decode)

    rows = [json.loads(line) for line in open(args.jsonl)
            if json.loads(line)['split'] == 'test']
    row = rows[args.row]
    body = tokenizer(row['body'], add_special_tokens=False)['input_ids']
    suffix = tokenizer(f"\n\nQuestion: {row['question']}",
                       add_special_tokens=False)['input_ids']
    ids = body + suffix
    end = len(body)

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
        umodel, gate, [ids], [0], [end], 1, args.k, prune_below_gate=True)
    for h in handles:
        h.remove()
    n_kept = n_kepts[0]
    upstream_kept = [p for p in kept_pos[0].tolist() if p >= 0][:n_kept]
    raw_in = captures['up_layer2'][0][0]
    if raw_in.dim() == 2:
        raw_in = raw_in[None]
    up_boundary = raw_in[:, :n_kept]                       # (1, n_kept, d)
    raw_out = captures['up_layer2'][0][1]
    if raw_out.dim() == 2:
        raw_out = raw_out[None]
    up_layer2 = raw_out[:, :n_kept]
    print(f'upstream kept {n_kept} positions; first token '
          f'{int(first_toks[0])} {tokenizer.decode([int(first_toks[0])])!r}')

    # ── our executor, forced kept set ─────────────────────────────────
    os.environ.update({'DAMENDAEMON_TOKEN_GATE': '1',
                       'DAMENDAEMON_TOKEN_GATE_LAYER': '1',
                       'DAMENDAEMON_TOKEN_GATE_K': str(args.k),
                       'DAMENDAEMON_TOKEN_GATE_WEIGHTS': args.gate})
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
    from damendaemon.token_gate import TokenGate
    model = load_model(args.merged, 24576)
    executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

    forced = torch.tensor([p for p in upstream_kept if 0 <= p < end],
                          dtype=torch.long, device=executor.device)
    all_positions = torch.arange(len(ids), device=executor.device)
    article_mask = torch.zeros(len(ids), dtype=torch.bool, device=executor.device)
    article_mask[0:end] = True
    kept_positions = torch.sort(torch.cat((all_positions[~article_mask], forced))).values

    slots = torch.tensor([0], device=executor.device)
    ids_t = torch.tensor([ids], device=executor.device)
    positions = torch.arange(len(ids), device=executor.device)[None]
    boundary_full = executor.forward_boundary(
        executor.embedding(ids_t), positions, slots, True)
    x = boundary_full.index_select(1, kept_positions)
    x = executor._mlp_half(x, 1)
    ours_boundary = x

    # our layer 2 on our boundary (the layerwise reproduction)
    print('[cross] computing our layer-2 forwards...', flush=True)
    h = executor.norm(ours_boundary, executor.w['2.norm1'])
    ours_layer2 = ours_boundary + executor.linear_attention(h, 2, slots, True, 0)
    print('[cross] ours-on-ours done', flush=True)

    # our layer 2 on UPSTREAM's boundary
    print('[cross] chunk_scan active:', executor.chunk_scan is not None,
          '| metal_scan:', executor.metal_scan is not None, flush=True)
    h_up = executor.norm(up_boundary.to(ours_boundary.dtype), executor.w['2.norm1'])
    ours_layer2_on_up = up_boundary.to(ours_boundary.dtype) + \
        executor.linear_attention(h_up, 2, slots, True, 0)
    print('[cross] ours-on-upstream done', flush=True)

    # Same forward with the chunk kernel disabled — the torch token
    # recurrence fallback.  The #641 property suite ran tiny (<64-token)
    # sequences that always take this fallback, so the chunked kernel path
    # was never covered by those tests.
    saved_chunk_scan = executor.chunk_scan
    executor.chunk_scan = None
    h_up2 = executor.norm(up_boundary.to(ours_boundary.dtype), executor.w['2.norm1'])
    ours_layer2_fallback_on_up = up_boundary.to(ours_boundary.dtype) + \
        executor.linear_attention(h_up2, 2, slots, True, 0)
    executor.chunk_scan = saved_chunk_scan
    print('[cross] fallback-on-upstream done', flush=True)
    up_layer2_on_ours = None

    # ── per-position boundary profile ─────────────────────────────────
    per_row = ((ours_boundary.float() - up_boundary.float()).norm(dim=-1)
               / up_boundary.float().norm(dim=-1).clamp_min(1e-9))[0]
    kept_list = kept_positions.tolist()
    article_rows = [j for j, p in enumerate(kept_list) if p < end]
    suffix_rows = [j for j, p in enumerate(kept_list) if p >= end]
    def bucket(js):
        return float(per_row[js].mean()) if js else float('nan')
    profile = dict(overall=float(per_row.mean()),
                   article_mean=bucket(article_rows),
                   suffix_mean=bucket(suffix_rows),
                   first100=bucket(article_rows[:100]),
                   last100_art=bucket(article_rows[-100:]),
                   max_row=float(per_row.max()),
                   argmax_row=int(per_row.argmax()))

    report = dict(n_kept=n_kept,
                  boundary_rms=relrms(ours_boundary, up_boundary),
                  layer2_rms=relrms(ours_layer2, up_layer2),
                  ours_layer2_on_up_boundary=relrms(ours_layer2_on_up, up_layer2),
                  fallback_layer2_on_up_boundary=relrms(ours_layer2_fallback_on_up, up_layer2),
                  chunk_scan_active=executor.chunk_scan is not None,
                  up_layer2_on_ours_boundary=None,
                  up_layer2_on_ours_vs_ours=None,
                  profile=profile)
    print(json.dumps(report, indent=1))
    torch.save(dict(ours_boundary=ours_boundary, up_boundary=up_boundary,
                    ours_layer2=ours_layer2, up_layer2=up_layer2,
                    ours_layer2_on_up=ours_layer2_on_up,
                    up_layer2_on_ours=up_layer2_on_ours,
                    kept_positions=kept_positions), args.dump)
    json.dump(report, open(args.out, 'w'), indent=1)
    print('wrote', args.out, 'and', args.dump)


if __name__ == '__main__':
    main()
