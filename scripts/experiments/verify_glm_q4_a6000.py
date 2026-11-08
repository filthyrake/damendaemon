"""Bounded A6000 cell: GLM-5.3 Q4_0 expert bank vs shipped-FP8 oracle (#719 rung 2).

Loads one real sparse-MoE layer (default 6) of the official GLM-5.3 checkpoint
from the staged NAS original, builds TWO resident CUDA banks through the
qualified executor path:

- oracle: the shipped blockwise-E4M3-FP8 storage placed directly
  (CudaPackedMatrix format 'e4m3fn_f32' — byte-identical to the checkpoint's
  uint8 codes + FP32 128x128 scale_inv grids)
- candidate: the same weights packed to canonical Q4_0 on device
  (damendaemon.glm_q4 decode + quantization.quantize_q4), admitted by the
  widened CUDA GLM bank admission (uniform 'q4_0_f16' bank)

Both banks execute through the production CudaPackedGlmFeedForward with the
REAL router and correction bias (high precision, never packed — #668).

Pre-declared gates (on #719 before launch; drift envelope derives from rung 1's
measured weight-level q4-of-FP8 drift, worst 10.24% rel RMS — the gate is 1.5x
that family; no token or quality claim, no throughput claim):

1. PER-EXPERT PARITY: for every expert selected by the real router on the
   pinned inputs plus the shared expert, candidate-expert output vs
   oracle-expert output rel RMS <= 0.15.
2. FFN PARITY: full routed FFN output (same router, same inputs, batched
   composition) rel RMS <= 0.15 and max_abs <= 0.15 x oracle absmax.
3. COMPOSITION EQUIVALENCE inside the q4 bank: batched vs per-token output
   rtol 3e-5 / atol 3e-6 (identical per-row arithmetic by construction).
4. MEMORY: q4 bank bytes < fp8 bank bytes; peak reserved recorded. No
   performance or SLO claim anywhere.
"""
import argparse
import json
from pathlib import Path

from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--tokens', type=int, default=8)
    args = parser.parse_args()
    require_guard()
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    from safetensors import safe_open
    from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward
    from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
    from damendaemon.glm_q4 import decode_blockwise_fp8, projected_q4_bank_bytes
    from damendaemon.quantization import quantize_q4

    if torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(int(26 * 2**30) / properties.total_memory)
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  layer=args.layer, model_qualified=False, cases=[], gates=[])
    save(args.output, report)
    try:
        device = torch.device('cuda:0')
        layer = args.layer
        index = json.load(open(f'{args.model}/model.safetensors.index.json'))['weight_map']

        def names_for(expert):
            prefix = f'model.layers.{layer}.mlp.' + ('shared_experts' if expert == 'shared' else f'experts.{expert}')
            return {p: [f'{prefix}.{p}_proj.weight', f'{prefix}.{p}_proj.weight_scale_inv']
                    for p in ('gate', 'up', 'down')}
        router_names = [f'model.layers.{layer}.mlp.gate.weight',
                        f'model.layers.{layer}.mlp.gate.e_score_correction_bias']
        wanted = router_names
        experts = 0
        while f'model.layers.{layer}.mlp.experts.{experts}.gate_proj.weight_scale_inv' in index:
            experts += 1
        if experts < 2:
            raise ValueError(f'layer {layer} is not a sparse-MoE layer')
        for expert in range(experts):
            for pair in names_for(expert).values():
                wanted.extend(pair)
        for pair in names_for('shared').values():
            wanted.extend(pair)
        missing = [n for n in wanted if n not in index]
        if missing:
            raise ValueError(f'checkpoint misses {missing[:3]}')

        by_shard = {}
        for name in wanted:
            by_shard.setdefault(index[name], []).append(name)
        tensors = {}
        for shard, names in by_shard.items():
            with safe_open(f'{args.model}/{shard}', framework='pt', device='cpu') as f:
                for name in names:
                    tensors[name] = f.get_tensor(name)
        report.update(experts=experts, shards=len(by_shard))

        # Router and bias stay high precision (#668): record the shipped dtype.
        router = tensors[router_names[0]].float().to(device)
        bias = tensors[router_names[1]].float().to(device)
        report['router_source_dtype'] = str(tensors[router_names[0]].dtype)

        def gpu_matrix(prefix_part):
            # safetensors hands back native float8_e4m3fn codes; the packed-tile
            # decoder reads uint8 byte storage — same bytes, one view.
            raw = tensors[f'{prefix_part}_proj.weight']
            if raw.dtype == torch.uint8:
                codes = raw
            elif raw.dtype == torch.float8_e4m3fn:
                codes = raw.view(torch.uint8)
            else:
                raise ValueError(f'unexpected shipped codes dtype {raw.dtype} for {prefix_part}')
            scales = tensors[f'{prefix_part}_proj.weight_scale_inv']
            if scales.dtype != torch.float32:
                raise ValueError(f'unexpected shipped scale dtype {scales.dtype} for {prefix_part}')
            return codes.to(device), scales.to(device)

        oracle_bank, oracle_shared = [], None
        for expert in range(experts):
            matrices = [CudaPackedMatrix(*gpu_matrix(f'model.layers.{layer}.mlp.experts.{expert}.{p}'), 'e4m3fn_f32')
                        for p in ('gate', 'up', 'down')]
            oracle_bank.append(CudaPackedExpert(*matrices))
        oracle_shared = CudaPackedExpert(*[CudaPackedMatrix(*gpu_matrix(f'model.layers.{layer}.mlp.shared_experts.{p}'), 'e4m3fn_f32')
                                           for p in ('gate', 'up', 'down')])

        q4_bank, q4_shared = [], None

        def pack_q4(codes, scales):
            truth = decode_blockwise_fp8(codes, scales)
            weight, block_scales = quantize_q4(truth)
            return CudaPackedMatrix(weight, block_scales, 'q4_0_f16')

        for expert in range(experts):
            matrices = [pack_q4(*gpu_matrix(f'model.layers.{layer}.mlp.experts.{expert}.{p}'))
                        for p in ('gate', 'up', 'down')]
            q4_bank.append(CudaPackedExpert(*matrices))
        q4_shared = CudaPackedExpert(*[pack_q4(*gpu_matrix(f'model.layers.{layer}.mlp.shared_experts.{p}'))
                                      for p in ('gate', 'up', 'down')])
        fp8_bytes = sum(m.weight.numel() + m.scales.numel() * 4
                        for e in oracle_bank + [oracle_shared] for m in (e.gate, e.up, e.down))
        q4_bytes = sum(m.weight.numel() + m.scales.numel() * 2
                       for e in q4_bank + [q4_shared] for m in (e.gate, e.up, e.down))
        report.update(fp8_bank_bytes=fp8_bytes, q4_bank_bytes=q4_bytes,
                      q4_projected_bytes=projected_q4_bank_bytes(
                          [(m.weight.shape[0], m.weight.shape[1] * 2) for e in q4_bank for m in (e.gate, e.up, e.down)]))
        save(args.output, report)

        # Gate 4 first (memory shape of the cell itself).
        if not q4_bytes < fp8_bytes:
            raise ValueError('q4 bank must be smaller than the fp8 bank')
        report['gates'].append(dict(gate='q4_smaller_than_fp8', passed=True,
                                    ratio=round(q4_bytes / fp8_bytes, 4)))
        save(args.output, report)

        rng = torch.Generator(device='cpu').manual_seed(719)
        hidden = torch.randn(1, args.tokens, oracle_bank[0].width, generator=rng).to(device)
        oracle_ffn = CudaPackedGlmFeedForward(oracle_bank, router=router, correction_bias=bias,
                                               shared=oracle_shared, top_k=8, composition='batched')
        q4_ffn = CudaPackedGlmFeedForward(q4_bank, router=router, correction_bias=bias,
                                           shared=q4_shared, top_k=8, composition='batched')

        # Gate 1: per-expert parity on the experts the real router selects.
        logits = torch.nn.functional.linear(hidden.reshape(-1, hidden.shape[-1]), router)
        from damendaemon.cuda_glm_ffn import _select
        coefficients, ids = _select(logits, bias, 8, 2.5)
        selected = sorted(set(ids.reshape(-1).tolist()))
        rows = hidden.reshape(-1, hidden.shape[-1])
        expert_gates = []
        for expert_id in selected:
            oracle_expert = oracle_ffn.experts[expert_id].forward_batched(rows)
            q4_expert = q4_ffn.experts[expert_id].forward_batched(rows)
            delta = (q4_expert - oracle_expert).float()
            rel = float((delta.norm() / oracle_expert.float().norm()).item())
            entry = dict(expert=expert_id, rel_rms=rel, passed=rel <= 0.15)
            expert_gates.append(entry)
        oracle_shared_out = oracle_ffn.shared.forward_batched(rows)
        q4_shared_out = q4_ffn.shared.forward_batched(rows)
        shared_rel = float(((q4_shared_out - oracle_shared_out).norm() / oracle_shared_out.norm()).item())
        expert_gates.append(dict(expert='shared', rel_rms=shared_rel, passed=shared_rel <= 0.15))
        worst = max(e['rel_rms'] for e in expert_gates)
        report['gates'].append(dict(gate='per_expert_parity', passed=all(e['passed'] for e in expert_gates),
                                    worst_rel_rms=worst, selected=selected, cases=expert_gates))
        save(args.output, report)

        # Gate 2: full FFN parity (same router, same inputs).
        oracle_out = oracle_ffn(hidden)
        q4_out = q4_ffn(hidden)
        delta = q4_out - oracle_out
        rel = float((delta.norm() / oracle_out.norm()).item())
        max_abs = float(delta.abs().max().item())
        absmax = float(oracle_out.abs().max().item())
        gate2 = dict(gate='ffn_parity', rel_rms=rel, max_abs=max_abs, oracle_absmax=absmax,
                    max_abs_of_absmax=max_abs / absmax)
        gate2['passed'] = rel <= 0.15 and max_abs <= 0.15 * absmax
        report['gates'].append(gate2)
        save(args.output, report)

        # Gate 3: composition equivalence inside the q4 bank.
        q4_per_token = CudaPackedGlmFeedForward(q4_bank, router=router, correction_bias=bias,
                                                shared=q4_shared, top_k=8, composition='per_token')(hidden)
        composition_gap = float((q4_per_token - q4_out).abs().max().item())
        report['gates'].append(dict(gate='q4_composition_equivalence', max_abs=composition_gap,
                                    passed=composition_gap <= 3e-6 + 3e-5 * float(q4_out.abs().max().item())))
        save(args.output, report)

        report.update(passed=all(g['passed'] for g in report['gates']),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      routing_ids=selected, tokens=args.tokens,
                      hidden_width=oracle_bank[0].width)
        save(args.output, report)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
