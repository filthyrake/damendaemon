"""Bounded HOMER (Radeon 8060S, gfx1151 via ROCm/HIP) cell: GLM-5.3 Q4_0 expert
bank vs shipped-FP8 oracle (#719 rung 3a — the CUDA→ROCm boundary's first half).

Same paired-oracle design as the A6000 rung-2 cell (verify_glm_q4_a6000),
scoped to a real subset of layer-6 experts (default 48 of 256) with the real
router rows and correction bias subset to match, plus the full shared expert:
the 8060S is an APU whose compute shares die/memory with the host, so the
bring-up cell stays deliberately small. Pure-torch packed execution — no
native CUDA libraries (rocm_policy refuses those; this path never touches
them) — PyTorch's HIP build presents its devices through the CUDA API, so
the qualified CudaPacked* machinery runs unmodified.

Pre-declared gates (on #719 before launch — same envelope derivation as rung 2:
weight-level q4-of-FP8 drift worst 10.24% rel RMS through 3 projections +
SwiGLU; gates at 1.5x that family):

1. PER-EXPERT PARITY (routed experts selected on the pinned inputs + shared):
   rel RMS <= 0.15.
2. FFN PARITY: rel RMS <= 0.15 and max_abs <= 0.15 x oracle absmax.
3. COMPOSITION EQUIVALENCE inside the q4 bank: batched vs per-token,
   rtol 3e-5 / atol 3e-6.
4. MEMORY: q4 bank bytes < fp8 bank bytes; peak reserved recorded.

No token, quality, throughput or SLO claim. Thermal envelope: fans fixed/5
(RPM floors verified), guard_homer default limit 80 C; any trip pauses the lane.
"""
import argparse
import json
from pathlib import Path

from scripts.experiments.split_gguf import save


def require_homer_cooling(report):
    """Fail-closed fan/thermal admission inside the cell (the #537 lane pattern).

    Fans must be fixed level 5 with verified RPM readback and every
    k10temp/amdgpu sensor under the 80 C limit — CoolingGuard is the same
    fail-closed scan guard_homer runs; the outer guard remains the kill
    switch, this refuses to start without the prerequisites.
    """
    from scripts.experiments.homer_cooling import CoolingGuard
    guard = CoolingGuard(report, limit_celsius=80)
    guard.check(startup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/mnt/nas-models/models/GLM-5.3')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--experts', type=int, default=48)
    parser.add_argument('--tokens', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new report path required')
    import torch
    from safetensors import safe_open
    from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward, _select
    from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
    from damendaemon.glm_q4 import decode_blockwise_fp8, projected_q4_bank_bytes
    from damendaemon.quantization import quantize_q4

    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('exactly one HIP (ROCm) GPU required')
    if torch.cuda.get_device_name().startswith('NVIDIA'):
        raise ValueError('this cell targets the AMD 8060S; refusing NVIDIA')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(int(10 * 2**30) / properties.total_memory)
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  hip=getattr(torch.version, 'hip', None), layer=args.layer,
                  experts=args.experts, model_qualified=False, cases=[], gates=[])
    require_homer_cooling(report)
    save(args.output, report)
    try:
        device = torch.device('cuda:0')
        layer = args.layer
        index = json.load(open(f'{args.model}/model.safetensors.index.json'))['weight_map']
        total = 0
        while f'model.layers.{layer}.mlp.experts.{total}.gate_proj.weight_scale_inv' in index:
            total += 1
        if total < 2 or not 2 <= args.experts <= total:
            raise ValueError(f'layer {layer}: {total} experts; subset must be 2..{total}')

        wanted = [f'model.layers.{layer}.mlp.gate.weight',
                  f'model.layers.{layer}.mlp.gate.e_score_correction_bias']
        for expert in list(range(args.experts)) + ['shared_experts']:
            prefix = (f'model.layers.{layer}.mlp.experts.{expert}' if expert != 'shared_experts'
                      else f'model.layers.{layer}.mlp.shared_experts')
            for part in ('gate', 'up', 'down'):
                for suffix in ('weight', 'weight_scale_inv'):
                    wanted.append(f'{prefix}.{part}_proj.{suffix}')
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
        report.update(total_experts=total, shards=len(by_shard))

        router_full = tensors[f'model.layers.{layer}.mlp.gate.weight']
        bias_full = tensors[f'model.layers.{layer}.mlp.gate.e_score_correction_bias']
        router = router_full[:args.experts].float().to(device)
        bias = bias_full[:args.experts].float().to(device)
        report['router_source_dtype'] = str(router_full.dtype)

        def gpu_matrix(expert, part):
            prefix = (f'model.layers.{layer}.mlp.shared_experts' if expert == 'shared'
                     else f'model.layers.{layer}.mlp.experts.{expert}')
            raw = tensors[f'{prefix}.{part}_proj.weight']
            if raw.dtype == torch.uint8:
                codes = raw
            elif raw.dtype == torch.float8_e4m3fn:
                codes = raw.view(torch.uint8)
            else:
                raise ValueError(f'unexpected shipped codes dtype {raw.dtype}')
            scales = tensors[f'{prefix}.{part}_proj.weight_scale_inv']
            if scales.dtype != torch.float32:
                raise ValueError(f'unexpected shipped scale dtype {scales.dtype}')
            return codes.to(device), scales.to(device)

        oracle_bank, q4_bank, oracle_shared, q4_shared = [], [], None, None
        for expert in range(args.experts):
            oracle_bank.append(CudaPackedExpert(*[CudaPackedMatrix(*gpu_matrix(expert, p), 'e4m3fn_f32')
                                                for p in ('gate', 'up', 'down')]))
        oracle_shared = CudaPackedExpert(*[CudaPackedMatrix(*gpu_matrix('shared', p), 'e4m3fn_f32')
                                          for p in ('gate', 'up', 'down')])

        def pack_q4(codes, scales):
            truth = decode_blockwise_fp8(codes, scales)
            weight, block_scales = quantize_q4(truth)
            return CudaPackedMatrix(weight, block_scales, 'q4_0_f16')

        for expert in range(args.experts):
            q4_bank.append(CudaPackedExpert(*[pack_q4(*gpu_matrix(expert, p)) for p in ('gate', 'up', 'down')]))
        q4_shared = CudaPackedExpert(*[pack_q4(*gpu_matrix('shared', p)) for p in ('gate', 'up', 'down')])

        def bank_bytes(bank, shared, scale_width):
            total_bytes = 0
            for e in list(bank) + [shared]:
                for m in (e.gate, e.up, e.down):
                    total_bytes += m.weight.numel() + m.scales.numel() * scale_width
            return total_bytes

        fp8_bytes = bank_bytes(oracle_bank, oracle_shared, 4)
        q4_bytes = bank_bytes(q4_bank, q4_shared, 2)
        report.update(fp8_bank_bytes=fp8_bytes, q4_bank_bytes=q4_bytes,
                      q4_projected_bytes=projected_q4_bank_bytes(
                          [(m.weight.shape[0], m.weight.shape[1] * 2) for e in q4_bank
                           for m in (e.gate, e.up, e.down)]))
        save(args.output, report)

        if not q4_bytes < fp8_bytes:
            raise ValueError('q4 bank must be smaller than the fp8 bank')
        report['gates'].append(dict(gate='q4_smaller_than_fp8', passed=True,
                                    ratio=round(q4_bytes / fp8_bytes, 4)))
        save(args.output, report)

        rng = torch.Generator().manual_seed(719)
        hidden = torch.randn(1, args.tokens, oracle_bank[0].width, generator=rng).to(device)
        oracle_ffn = CudaPackedGlmFeedForward(oracle_bank, router=router, correction_bias=bias,
                                               shared=oracle_shared, top_k=min(8, args.experts),
                                               composition='batched')
        q4_ffn = CudaPackedGlmFeedForward(q4_bank, router=router, correction_bias=bias,
                                          shared=q4_shared, top_k=min(8, args.experts),
                                          composition='batched')
        logits = torch.nn.functional.linear(hidden.reshape(-1, hidden.shape[-1]), router)
        coefficients, ids = _select(logits, bias, min(8, args.experts), 2.5)
        selected = sorted(set(ids.reshape(-1).tolist()))
        rows = hidden.reshape(-1, hidden.shape[-1])
        expert_gates = []
        for expert_id in selected:
            oracle_expert = oracle_ffn.experts[expert_id].forward_batched(rows)
            q4_expert = q4_ffn.experts[expert_id].forward_batched(rows)
            rel = float(((q4_expert - oracle_expert).norm() / oracle_expert.norm()).item())
            expert_gates.append(dict(expert=expert_id, rel_rms=rel, passed=rel <= 0.15))
        oracle_shared_out = oracle_ffn.shared.forward_batched(rows)
        q4_shared_out = q4_ffn.shared.forward_batched(rows)
        shared_rel = float(((q4_shared_out - oracle_shared_out).norm() / oracle_shared_out.norm()).item())
        expert_gates.append(dict(expert='shared', rel_rms=shared_rel, passed=shared_rel <= 0.15))
        report['gates'].append(dict(gate='per_expert_parity', passed=all(e['passed'] for e in expert_gates),
                                    worst_rel_rms=max(e['rel_rms'] for e in expert_gates),
                                    selected=selected, cases=expert_gates))
        save(args.output, report)

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

        q4_per_token = CudaPackedGlmFeedForward(q4_bank, router=router, correction_bias=bias,
                                                shared=q4_shared, top_k=min(8, args.experts),
                                                composition='per_token')(hidden)
        gap = float((q4_per_token - q4_out).abs().max().item())
        report['gates'].append(dict(gate='q4_composition_equivalence', max_abs=gap,
                                    passed=gap <= 3e-6 + 3e-5 * float(q4_out.abs().max().item())))
        save(args.output, report)

        report.update(passed=all(g['passed'] for g in report['gates']),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      routing_ids=selected, tokens=args.tokens,
                      hidden_width=oracle_bank[0].width, subset_experts=args.experts)
        save(args.output, report)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
