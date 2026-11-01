#!/usr/bin/env python3
"""#668 real-weights CPU bring-up cell: ingest the pinned Qwen3.6-35B-A3B
safetensors snapshot through the new qwen3_5_moe path.

Cells (all CPU, no executor forward — the 67 GB FP16 checkpoint exceeds the
44.5 GiB A6000 and quantized MoE loads are cell 2):
  1. contract audit of the real snapshot (audit_snapshot: names, shapes,
     dtypes across all shards) against the transcribed fixture;
  2. full load through ModelWeights (config decode + audit + loader, every
     tensor consumed);
  3. real-weights compute smoke through the pinned host reference: routing
     distribution sanity on the real router (finite softmax, distinct top-8,
     renormalized weights) and a finite expert/shared FFN output.

Convention: ingestion/numerical-sanity evidence only; no quality, context,
concurrency or performance claim. Run on rocky10-desktop (124 GB RAM).
"""
if not __debug__:
    raise RuntimeError('run under normal python, without -O')

import argparse
import json
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/damen/models/Qwen_Qwen3.6-35B-A3B')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--out', default='qwen36-35b-cpu-bringup.json')
    args = parser.parse_args()

    sys.path.insert(0, '/tmp/668')
    report = {'model': args.model, 'context': args.context}

    # 1. contract audit over every shard header
    from damendaemon.hybrid_moe_checkpoint import audit_snapshot
    audit = audit_snapshot(args.model)
    report['audit'] = audit
    print(f'[audit] {audit}', flush=True)

    # 2. full load through ModelWeights
    t0 = time.perf_counter()
    from damendaemon.models import load as load_model
    model = load_model(args.model, args.context)
    load_seconds = time.perf_counter() - t0
    c = model.config
    report['config'] = {k: c[k] for k in ('model_type', 'num_hidden_layers', 'hidden_size',
                                          'num_experts', 'num_experts_per_tok',
                                          'moe_intermediate_size',
                                          'shared_expert_intermediate_size', 'head_dim',
                                          'num_attention_heads', 'num_key_value_heads')}
    report['layer_types'] = ''.join('L' if k == 'linear_attention' else 'F'
                                    for k in c['layer_types'])
    report['load_seconds'] = load_seconds
    report['weight_bytes'] = model.bytes
    print(f'[load] {load_seconds:.1f}s, {model.bytes/2**30:.1f} GiB; '
          f'{report["layer_types"]}; {c["num_experts"]} experts, top '
          f'{c["num_experts_per_tok"]}', flush=True)

    # 3. real-weights compute smoke through the pinned host reference
    from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward
    from damendaemon.expert_selection import select_experts
    import torch.nn.functional as F
    rng = torch.Generator().manual_seed(11)
    router = model.weights['1.router'].float()
    hidden = torch.randn(9, c['hidden_size'], generator=rng) * .5
    logits = F.linear(hidden, router)
    probabilities = logits.softmax(-1)
    coefficients, selected = select_experts(logits, c['num_experts_per_tok'],
                                            scoring='softmax', renormalize=True)
    report['routing'] = {
        'probabilities_finite': bool(torch.isfinite(probabilities).all()),
        'topk_distinct': bool(all(len(set(row.tolist())) == c['num_experts_per_tok']
                                  for row in selected)),
        'renormalized_sums': [round(float(row.sum()), 6) for row in coefficients[:3]],
        'expert_histogram_top3': [int(row) for row in torch.bincount(
            selected.flatten(), minlength=c['num_experts']).topk(3).indices],
    }
    print(f'[routing] {report["routing"]}', flush=True)

    reference = HybridMoeFeedForward(
        model.weights['1.router'].float(),
        model.weights['1.experts_gate_up'].float(),
        model.weights['1.experts_down'].float(),
        model.weights['1.shared_gate_up'][:c['shared_expert_intermediate_size']].float(),
        model.weights['1.shared_gate_up'][c['shared_expert_intermediate_size']:].float(),
        model.weights['1.shared_down'].float(),
        model.weights['1.shared_expert_gate'].float(),
        top_k=c['num_experts_per_tok'])
    output = reference.forward(hidden[None])
    report['ffn_smoke'] = {
        'finite': bool(torch.isfinite(output).all()),
        'output_rms': float(output.square().mean().sqrt()),
    }
    print(f'[ffn] layer-1 real-weights FFN output RMS '
          f'{report["ffn_smoke"]["output_rms"]:.4f}, finite '
          f'{report["ffn_smoke"]["finite"]}', flush=True)

    json.dump(report, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
