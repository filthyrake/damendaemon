#!/usr/bin/env python3
"""#668 real-weights CPU bring-up cell: load the pinned Qwen3.6-35B-A3B
safetensors snapshot through the new qwen3_5_moe path and run a bounded
numerical/decode check.

Cells:
  1. load + config decode + weight inventory (audit_names against the
     transcribed contract);
  2. tiny prefill + greedy decode vs the independent host reference for
     layer 1's FFN on real weights (routing parity on the actual router);
  3. short greedy generation sanity (finite logits, plausible text).

Convention: accuracy/drift comparison per the #23 pattern; no performance
claim (CPU FP32/FP16 reference speed is irrelevant). Run on rocky10-desktop
(124 GB RAM) with the staged tree.
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

    # 1. load through the full ModelWeights path (config + audit + loader)
    t0 = time.perf_counter()
    from damendaemon.models import load as load_model
    from damendaemon.hybrid import HybridExecutor
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
    print(f'[load] {load_seconds:.1f}s; {report["layer_types"]}; '
          f'{c["num_experts"]} experts, top {c["num_experts_per_tok"]}', flush=True)

    with torch.inference_mode():
        executor = HybridExecutor(model, use_graph=False, max_batch=1, kernels='torch')

        # 2. real-weights FFN parity on layer 1 (a linear-attention layer):
        # executor _moe_half vs the pinned host reference on the real router.
        from damendaemon.hybrid_moe_ffn import HybridMoeFeedForward
        rng = torch.Generator().manual_seed(11)
        reference = HybridMoeFeedForward(
            model.weights['1.router'].float(),
            model.weights['1.experts_gate_up'].float(),
            model.weights['1.experts_down'].float(),
            model.weights['1.shared_gate_up'][:c['shared_expert_intermediate_size']].float(),
            model.weights['1.shared_gate_up'][c['shared_expert_intermediate_size']:].float(),
            model.weights['1.shared_down'].float(),
            model.weights['1.shared_expert_gate'].float(),
            top_k=c['num_experts_per_tok'])
        x = (torch.randn(1, 7, c['hidden_size'], generator=rng) * .3)
        normed = executor.norm(x, executor.w['1.norm2']).float()
        expected = reference.forward(normed)
        actual = (executor._moe_half(x, 1) - x).reshape(normed.shape)
        rms = float((actual - expected).norm() / expected.norm().clamp_min(1e-9))
        report['ffn_parity_rms'] = rms
        print(f'[parity] layer-1 MoE FFN vs host reference: rel-RMS {rms:.3e}', flush=True)

        # 3. tiny prefill + greedy decode
        t1 = time.perf_counter()
        prompt = model.raw_tokens('The capital of France is')
        logits = executor.prefill(prompt)
        prefill_seconds = time.perf_counter() - t1
        generated = []
        token = int(logits.argmax(-1)[0])
        for _ in range(24):
            generated.append(token)
            logits, nxt = executor.decode(token)
            token = int(nxt[0])
        text = model.tokenizer.decode(generated)
        report.update(prefill_seconds=prefill_seconds, prompt_tokens=len(prompt),
                      generated_text=text[:200], finite=bool(torch.isfinite(logits).all()))
        print(f'[decode] prompt {len(prompt)} tok, prefill {prefill_seconds:.1f}s', flush=True)
        print('[decode] generated:', repr(text[:200]), flush=True)

    json.dump(report, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
