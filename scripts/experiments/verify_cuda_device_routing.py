"""Device-resident expert routing on the shared A2 versus the CPU selection control.

Tiny correctness fixture: routing semantics, deterministic ties, no host
readback, and direct consumption by prepared packed experts. Not a performance
measurement; shared-GPU timings are not qualification evidence.
"""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); require_guard()
    if args.output.exists(): raise ValueError('new output path required')
    import torch
    from dllm.cuda_expert_routing import select_experts_device
    from dllm.expert_selection import select_experts
    from tests.test_packed_moe import expert, dense_expert
    if torch.version.hip or torch.cuda.device_count() != 1: raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(64*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=torch.__version__,
                  cases=[], model_qualified=False, performance_qualified=False)
    save(args.output, report)
    try:
        rng = torch.Generator().manual_seed(478)
        logits_cpu = torch.randn(6, 8, generator=rng)
        logits_cpu[0, :] = logits_cpu[0, 2]  # one fully tied row: device ties prefer lower IDs
        logits = logits_cpu.cuda()
        bias = torch.linspace(-.2, .2, 8)
        bias_dev = bias.cuda()
        cases = [dict(), dict(renormalize=True), dict(scoring='sigmoid', renormalize=True),
                 dict(scoring='sqrtsoftplus'), dict(scale=3), dict(group_count=2, groups_to_keep=1),
                 dict(group_count=2, correction_bias='bias', renormalize=True)]
        from unittest.mock import patch
        for case in cases:
            device_case = {k: (bias_dev if v == 'bias' else v) for k, v in case.items()}
            host_case = {k: (bias if v == 'bias' else v) for k, v in case.items()}
            with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.tolist', side_effect=AssertionError('host readback')):
                weights, selected = select_experts_device(logits, 2, **device_case)
            if weights.device != logits.device or selected.dtype != torch.int64:
                raise AssertionError('routes left the routing device')
            expected_w, expected_s = select_experts(logits_cpu, 2, **host_case)
            torch.testing.assert_close(weights.cpu(), expected_w)
            if case.get('group_count', 1) > 1:
                if not torch.equal(selected.cpu(), expected_s):
                    raise AssertionError('grouped selection mismatch')
            else:
                # Plain top-k: values must match; tie order follows the device
                # lower-ID contract, so compare sorted values and control IDs on
                # the tie-free rows only.
                torch.testing.assert_close(weights.cpu().sort(-1, descending=True).values, expected_w)
                tied = (logits_cpu == logits_cpu[:, :1])
                if not bool((selected.cpu()[~tied.any(-1)] == expected_s[~tied.any(-1)]).all()):
                    raise AssertionError('plain selection mismatch on tie-free rows')
            report['cases'].append(dict(case={k: (v if not torch.is_tensor(v) else 'tensor') for k, v in case.items()},
                                        matched=True))
            save(args.output, report)
        # Fully tied row: device IDs must be 0,1 with equal weights.
        weights, selected = select_experts_device(logits[:1], 2)
        if not torch.equal(selected[0].cpu(), torch.tensor([0, 1])):
            raise AssertionError('device tie contract violated')
        report['tied_row_ids'] = [0, 1]
        # Prepared packed experts consume the device routes directly.
        from dllm.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
        from dllm.cuda_packed_moe import cuda_packed_moe
        controls = [expert('e2m1'), expert('e4m3fn'), expert('e2m1'), expert('e4m3fn'),
                    expert('e2m1'), expert('e4m3fn'), expert('e2m1'), expert('e4m3fn')]
        bank = []
        for original, _ in controls[:8]:
            matrices = [CudaPackedMatrix(m.weight.cuda(), m.scales.cuda(), m.format)
                        for m in (original.gate, original.up, original.down)]
            bank.append(CudaPackedExpert(*matrices, row_chunk=13, column_chunk=17, activation_limit=2))
        hidden = (torch.randn(6, 32, generator=rng)*.2).cuda()
        weights, selected = select_experts_device(logits, 2)
        result = cuda_packed_moe(hidden, bank, selected, weights)
        expected = torch.zeros(6, 32, device='cuda')
        for row in range(6):
            for route in range(2):
                dense = controls[int(selected[row, route])][1]
                host_w = weights[row:row+1, route:route+1]
                expected[row:row+1] += dense_expert(hidden[row:row+1].cpu(), dense, host_w.cpu()).float().cuda()
        torch.testing.assert_close(result, expected.to(result.dtype), rtol=2e-4, atol=1e-5)
        report['expert_consumed'] = True
        # Batched composition: same routes through forward_batched groups.
        batched_bank = []
        for original, _ in controls:
            matrices = [CudaPackedMatrix(m.weight.cuda(), m.scales.cuda(), m.format)
                        for m in (original.gate, original.up, original.down)]
            batched_bank.append(CudaPackedExpert(*matrices, row_chunk=13, column_chunk=17, activation_limit=2))
        batched_cases = 0
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            cast_hidden = hidden.to(dtype)
            for shared in (None, batched_bank[0]):
                per_token = cuda_packed_moe(cast_hidden, batched_bank, selected, weights, shared=shared)
                batched = cuda_packed_moe(cast_hidden, batched_bank, selected, weights, shared=shared, batched=True)
                # Duplicate routes accumulate atomically on CUDA: tolerance only.
                torch.testing.assert_close(batched, per_token, rtol=2e-4, atol=1e-5)
                batched_cases += 1
        empty_per = cuda_packed_moe(hidden[:0], batched_bank, selected[:0], weights[:0], shared=batched_bank[0])
        empty_bat = cuda_packed_moe(hidden[:0], batched_bank, selected[:0], weights[:0], shared=batched_bank[0], batched=True)
        if empty_per.shape != (0, 32) or empty_bat.shape != (0, 32): raise AssertionError('empty batched result')
        batched_cases += 1
        report['batched_cases'] = batched_cases
        # Prepared FFN dispatch switch-over: device_routing vs host routing.
        from dllm.cuda_glm_ffn import CudaPackedGlmFeedForward
        from dllm.cuda_deepseek_ffn import CudaPackedDeepSeekFeedForward
        ffn_cases = 0
        glm_bank = []
        for index in range(4):
            matrices = []
            for rows, columns in ((64, 32), (64, 32), (32, 64)):
                raw = (torch.rand(rows, columns, generator=rng)*120).to(torch.uint8)
                scales = torch.full(((rows+127)//128, (columns+127)//128), .125)
                matrices.append(CudaPackedMatrix(raw.cuda(), scales.cuda(), 'e4m3fn_f32'))
            glm_bank.append(CudaPackedExpert(*matrices, row_chunk=13, column_chunk=17, activation_limit=2))
        router = (torch.randn(4, 32, generator=rng)*.1).cuda()
        bias = (torch.randn(4, generator=rng)*.1).cuda()
        hidden3 = (torch.randn(1, 3, 32, generator=rng)*.2).cuda()
        host_ffn = CudaPackedGlmFeedForward(glm_bank, router=router, correction_bias=bias,
                                            shared=glm_bank[0], top_k=2, device_routing=False,
                                            activation_limit=2)
        device_ffn = CudaPackedGlmFeedForward(glm_bank, router=router, correction_bias=bias,
                                              shared=glm_bank[0], top_k=2, device_routing=True,
                                              activation_limit=2)
        torch.testing.assert_close(device_ffn(hidden3), host_ffn(hidden3), rtol=2e-4, atol=1e-5)
        batched_ffn = CudaPackedGlmFeedForward(glm_bank, router=router, correction_bias=bias,
                                               shared=glm_bank[0], top_k=2, device_routing=True,
                                               composition='batched', activation_limit=2)
        torch.testing.assert_close(batched_ffn(hidden3), host_ffn(hidden3), rtol=2e-4, atol=1e-5)
        ffn_cases += 1
        ds_bank = []
        for original, _ in controls[4:8]:
            matrices = [CudaPackedMatrix(m.weight.cuda(), m.scales.cuda(), m.format)
                        for m in (original.gate, original.up, original.down)]
            ds_bank.append(CudaPackedExpert(*matrices, row_chunk=13, column_chunk=17, activation_limit=2))
        ds_router = (torch.randn(4, 32, generator=rng)*.1).cuda()
        ds_bias = (torch.randn(4, generator=rng)*.1).cuda()
        ds_hidden = (torch.randn(1, 2, 32, generator=rng)*.2).cuda()
        ds_host = CudaPackedDeepSeekFeedForward(ds_bank, router=ds_router, shared=ds_bank[0],
                                                correction_bias=ds_bias, top_k=2, device_routing=False,
                                                activation_limit=2)
        ds_device = CudaPackedDeepSeekFeedForward(ds_bank, router=ds_router, shared=ds_bank[0],
                                                  correction_bias=ds_bias, top_k=2, device_routing=True,
                                                  activation_limit=2)
        torch.testing.assert_close(ds_device(ds_hidden), ds_host(ds_hidden), rtol=2e-4, atol=1e-5)
        ffn_cases += 1
        report['ffn_device_routing_cases'] = ffn_cases
        # Fused composition: no host sync, uniform-format bank, same route semantics.
        fused_bank = bank[1::2]  # four e4m3fn experts; routes map to 0..3
        fused_selected = selected % len(fused_bank)
        fused_cases = 0
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            cast_hidden = hidden.to(dtype)
            control = cuda_packed_moe(cast_hidden, fused_bank, fused_selected, weights,
                                      shared=fused_bank[0], batched=True)
            with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.tolist', side_effect=AssertionError('host readback')):
                fused = cuda_packed_moe(cast_hidden, fused_bank, fused_selected, weights,
                                        shared=fused_bank[0], fused=True)
            report['fused_debug'] = dict(
                dtype=str(dtype), shared=shared is not None,
                routes=fused_selected.tolist(),
                diff_positions=(fused.float()-control.float()).abs().gt(1e-4).nonzero().tolist(),
                fused_sample=[round(v, 5) for v in fused[1, 8:16].tolist()],
                control_sample=[round(v, 5) for v in control[1, 8:16].tolist()])
            save(args.output, report)
            # The fused bmm reduction order differs from the per-row matmul;
            # FP16/BF16 casts amplify the 1-ULP FP32 difference to one ULP.
            rtol, atol = (2e-4, 1e-5) if dtype == torch.float32 else (.02, .002)
            torch.testing.assert_close(fused, control, rtol=rtol, atol=atol)
            fused_cases += 1
        with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')):
            fused_empty = cuda_packed_moe(hidden[:0], fused_bank, fused_selected[:0], weights[:0],
                                          shared=fused_bank[0], fused=True)
        if fused_empty.shape != (0, 32): raise AssertionError('empty fused result')
        fused_cases += 1
        try:
            cuda_packed_moe(hidden, fused_bank, fused_selected, weights, fused=True, max_gather_bytes=1)
        except ValueError: pass
        else: raise AssertionError('oversized gather admitted')
        fused_cases += 1
        report['fused_cases'] = fused_cases
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['passed'] = True
    except BaseException as error:
        report['error'] = str(error); raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
