"""Prove paired activation bytes preserve both native terms and every scale."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
from pathlib import Path

import torch

from damendaemon.native_q4_residual import library


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--paired', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert torch.cuda.device_count() == 1
    torch.manual_seed(371)
    report = dict(passed=False, device=torch.cuda.get_device_name(), checks=[],
                  scope='Native quantizer byte-layout equivalence; no speed or model claim.',
                  libraries={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for path in (args.baseline, *args.paired)})
    def quantize(value, path):
        width = value.numel()
        first = torch.empty_like(value, dtype=torch.int8)
        second = torch.empty_like(first)
        scales = torch.empty((width // 32, 4), device='cuda', dtype=torch.float32)
        _, operation, _ = library(path)
        error = operation(value.data_ptr(), first.data_ptr(), second.data_ptr(), scales.data_ptr(),
                          1, width, torch.cuda.current_stream().cuda_stream)
        assert error == 0, error
        return first, second, scales
    def canonical(paired):
        return paired.reshape(-1, 4, 2, 4).permute(0, 2, 1, 3).reshape_as(paired)
    try:
        for width in (256, 4096, 14336):
            indices = torch.arange(width, device='cuda')
            signs = (1 - 2 * (indices % 2)).float()
            random = torch.randn(width, device='cuda')
            cases = dict(zero=torch.zeros_like(random), unit=signs,
                         subnormal=signs * 2.**-24, maximum=signs * 65504,
                         mixed_exponents=signs * torch.pow(2., (indices % 39 - 24).float()) * (1 + (indices % 7) * .125))
            for exponent in (-14, 0, 10):
                cases[f'random_{exponent}'] = random * 2.**exponent
            for name, value in cases.items():
                value = value.half().reshape(1, width)
                assert torch.isfinite(value).all()
                baseline = quantize(value, args.baseline)
                for path in args.paired:
                    first, second, scales = quantize(value, path)
                    exact = (torch.equal(canonical(first), baseline[0]) and
                             torch.equal(canonical(second), baseline[1]) and
                             torch.equal(scales, baseline[2]))
                    report['checks'].append(dict(width=width, case=name, paired_library=path, passed=exact))
                    assert exact, report['checks'][-1]
        report['passed'] = True
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
