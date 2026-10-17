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
        cases = [dict(), dict(renormalize=True), dict(scoring='sigmoid', renormalize=True),
                 dict(scoring='sqrtsoftplus'), dict(scale=3), dict(group_count=2, groups_to_keep=1),
                 dict(group_count=2, correction_bias=bias, renormalize=True)]
        from unittest.mock import patch
        for case in cases:
            with patch('torch.Tensor.item', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.cpu', side_effect=AssertionError('host readback')), \
                    patch('torch.Tensor.tolist', side_effect=AssertionError('host readback')):
                weights, selected = select_experts_device(logits, 2, **case)
            if weights.device != logits.device or selected.dtype != torch.int64:
                raise AssertionError('routes left the routing device')
            expected_w, expected_s = select_experts(logits_cpu, 2, **case)
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
        if not torch.equal(selected[0], torch.tensor([0, 1])):
            raise AssertionError('device tie contract violated')
        report['tied_row_ids'] = [0, 1]
        # Prepared packed experts consume the device routes directly.
        from dllm.cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix
        from dllm.cuda_packed_moe import cuda_packed_moe
        controls = [expert('e2m1'), expert('e4m3fn'), expert('e2m1'), expert('e4m3fn'),
                    expert('e2m1'), expert('e4m3fn'), expert('e2m1'), expert('e4m3fn')]
        bank = []
        for original, _ in controls:
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
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['passed'] = True
    except BaseException as error:
        report['error'] = str(error); raise
    finally:
        save(args.output, report)


if __name__ == '__main__':
    main()
