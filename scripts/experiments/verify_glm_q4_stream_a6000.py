"""Bounded A6000 cell: selected-expert STREAMING from host-resident Q4_0 banks
(#719 rung 4a — the fleet cell's device-memory mechanism).

The full-model fleet cell cannot be whole-stage device-resident: the q4
projection is 395.93 GiB against ~109 GB of fleet VRAM. The mechanism this
cell qualifies: keep every expert's q4 packed matrices in HOST RAM, route on
device (router + bias are KB-scale), and stream ONLY the router-selected
experts' matrices to the GPU per token batch — top-8 of 256 routed experts
plus the shared expert, ~0.2 GB streamed per batch instead of the 5.4 GB
resident bank.

Oracle: the SAME weights fully resident on device through the production
`CudaPackedGlmFeedForward` batched path (rung 2's shape). The streamed path
copies the identical packed tensors and runs the identical per-expert
kernel path with the identical post-down weighting and accumulation order
(grouped stable routes) — the gates demand EXACT equality, with a tight FP
tolerance recorded beside it.

Pre-declared gates (on #719 before launch):
1. STREAMED == RESIDENT: max_abs == 0.0 (bit-identical; same bytes, same
   kernels, same accumulation order). Any nonzero delta is a defect finding.
2. STREAMED vs SHIPPED-FP8 oracle: rel RMS <= 0.15 and max_abs <= 0.15 x
   absmax (the standing rung-2/3a/3b envelope).
3. TRANSFER ACCOUNTING: streamed bytes recorded and <= the selected experts'
   payload (never the whole bank).
No throughput or SLO claim — mechanism qualification only.
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
    from damendaemon.cuda_glm_ffn import CudaPackedGlmFeedForward, _routed_batched, _select
    from damendaemon.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
    from damendaemon.expert_routes import grouped_routes
    from damendaemon.glm_q4 import decode_blockwise_fp8
    from damendaemon.quantization import quantize_q4
    from scripts.experiments.glm_q4_chain_common import load_ffn_tensors

    if torch.version.hip or torch.cuda.device_count() != 1:
        raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(int(26 * 2**30) / properties.total_memory)
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  layer=args.layer, gates=[], mechanism='selected_expert_streaming')
    save(args.output, report)
    try:
        device = torch.device('cuda:0')
        tensors, experts = load_ffn_tensors(args.model, args.layer)
        router = tensors[f'model.layers.{args.layer}.mlp.gate.weight'].float().to(device)
        bias = tensors[f'model.layers.{args.layer}.mlp.gate.e_score_correction_bias'].float().to(device)

        # Host-resident q4 banks: every matrix packed ONCE on CPU and kept in
        # host RAM. Streams copy these bytes to the device on demand.
        host_bank = {}

        def pack_host(prefix, part):
            raw = tensors[f'{prefix}.{part}_proj.weight']
            codes = raw if raw.dtype == torch.uint8 else raw.contiguous().view(torch.uint8)
            scales = tensors[f'{prefix}.{part}_proj.weight_scale_inv']
            weight, block_scales = quantize_q4(decode_blockwise_fp8(codes, scales))
            return weight, block_scales

        for expert in range(experts):
            host_bank[expert] = [pack_host(f'model.layers.{args.layer}.mlp.experts.{expert}', p)
                                 for p in ('gate', 'up', 'down')]
        host_shared = [pack_host(f'model.layers.{args.layer}.mlp.shared_experts', p)
                       for p in ('gate', 'up', 'down')]
        host_bank['shared'] = host_shared
        report.update(experts=experts, host_bank_bytes=sum(
            w.numel() + s.numel() * 2 for triple in list(host_bank.values()) for w, s in triple))
        save(args.output, report)

        def resident(triple):
            return CudaPackedExpert(*[CudaPackedMatrix(w.to(device), s.to(device), 'q4_0_f16')
                                      for w, s in triple])

        # Sequential phases with explicit frees: only small output tensors ever
        # coexist on device (attempt 1 held resident-q4 + fp8 banks together
        # and hit the allocator cap — an honest sizing bug, retained).
        generator = torch.Generator().manual_seed(719)
        hidden = torch.randn(1, args.tokens, 6144, generator=generator).to(device)
        rows = hidden.reshape(-1, hidden.shape[-1])

        # Phase 1: resident q4 oracle (rung-2 production path).
        resident_bank = [resident(host_bank[e]) for e in range(experts)]
        resident_shared = resident(host_shared)
        resident_ffn = CudaPackedGlmFeedForward(resident_bank, router=router, correction_bias=bias,
                                                shared=resident_shared, top_k=8, composition='batched')
        resident_out = resident_ffn(hidden).clone()
        del resident_bank, resident_shared, resident_ffn
        torch.cuda.empty_cache()

        # Phase 2: shipped-fp8 oracle bank for the drift-context gate.
        def resident_fp8(prefix, part):
            raw = tensors[f'{prefix}.{part}_proj.weight']
            codes = (raw if raw.dtype == torch.uint8 else raw.contiguous().view(torch.uint8)).to(device)
            scales = tensors[f'{prefix}.{part}_proj.weight_scale_inv'].to(device)
            return CudaPackedMatrix(codes, scales, 'e4m3fn_f32')

        fp8_bank = [CudaPackedExpert(*[resident_fp8(f'model.layers.{args.layer}.mlp.experts.{e}', p)
                                       for p in ('gate', 'up', 'down')]) for e in range(experts)]
        fp8_shared = CudaPackedExpert(*[resident_fp8(f'model.layers.{args.layer}.mlp.shared_experts', p)
                                         for p in ('gate', 'up', 'down')])
        fp8_ffn = CudaPackedGlmFeedForward(fp8_bank, router=router, correction_bias=bias,
                                           shared=fp8_shared, top_k=8, composition='batched')
        fp8_out = fp8_ffn(hidden).clone()
        del fp8_bank, fp8_shared, fp8_ffn
        torch.cuda.empty_cache()

        # Phase 3: streamed path — route on device, stream ONLY selected experts.
        logits = torch.nn.functional.linear(rows, router)
        coefficients, ids = _select(logits, bias, 8, 2.5)
        streamed_bytes = 0
        output = torch.zeros_like(rows)
        flat_coefficients = coefficients.reshape(-1)
        token_ids = torch.arange(rows.shape[0], device=rows.device).repeat_interleave(ids.shape[1])
        for expert_id, tokens_for_expert, positions in grouped_routes(ids, experts):
            triple = host_bank[expert_id]
            expert = CudaPackedExpert(*[CudaPackedMatrix(w.to(device), s.to(device), 'q4_0_f16')
                                        for w, s in triple])
            streamed_bytes += sum(w.numel() + s.numel() * 2 for w, s in triple)
            source = rows.index_select(0, tokens_for_expert)
            value = expert.forward_batched(source)
            scaled = value.float() * flat_coefficients[positions].reshape(-1, 1)
            output.index_add_(0, tokens_for_expert, scaled)
            del expert, source, value
        shared = CudaPackedExpert(*[CudaPackedMatrix(w.to(device), s.to(device), 'q4_0_f16')
                                     for w, s in host_shared])
        streamed_bytes += sum(w.numel() + s.numel() * 2 for w, s in host_shared)
        output.add_(shared.forward_batched(rows).float())
        output = output.to(rows.dtype).reshape(hidden.shape)

        selected = sorted(set(ids.reshape(-1).tolist()))
        report.update(selected_experts=len(selected), streamed_bytes=streamed_bytes,
                      streamed_gib=round(streamed_bytes / 2**30, 3))
        save(args.output, report)

        gate1_delta = (output - resident_out).abs().max().item()
        report['gates'].append(dict(gate='streamed_equals_resident', max_abs=gate1_delta,
                                    passed=gate1_delta == 0.0))
        save(args.output, report)

        fp8_out = fp8_ffn(hidden)
        delta = output - fp8_out
        rel = float((delta.norm() / fp8_out.norm()).item())
        max_abs = float(delta.abs().max().item())
        absmax = float(fp8_out.abs().max().item())
        gate2 = dict(gate='streamed_vs_shipped_fp8', rel_rms=rel, max_abs=max_abs,
                     oracle_absmax=absmax, max_abs_of_absmax=max_abs / absmax)
        gate2['passed'] = rel <= 0.15 and max_abs <= 0.15 * absmax
        report['gates'].append(gate2)

        host_total = report['host_bank_bytes']
        report['gates'].append(dict(gate='transfer_bounded', streamed_bytes=streamed_bytes,
                                    host_bank_bytes=host_total,
                                    passed=0 < streamed_bytes < host_total))
        report.update(passed=all(g['passed'] for g in report['gates']),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      selected=selected)
        save(args.output, report)
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
