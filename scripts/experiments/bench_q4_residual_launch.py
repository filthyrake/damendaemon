"""Matched native Q4 launch experiments; no model/serving performance claim."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
from pathlib import Path
import signal
import statistics

import torch
import triton.testing

from dllm.native_q4_residual import linear
from dllm.quantization import dequantize_q4, quantize_q4


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', action='append', required=True, help='NAME=LIBRARY')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    candidates = dict(item.split('=', 1) for item in args.candidate)
    if len(candidates) != len(args.candidate) or 'baseline' in candidates:
        parser.error('candidate names must be unique and may not be baseline')
    assert torch.cuda.device_count() == 1, 'select exactly one GPU'
    torch.set_num_threads(4)
    torch.manual_seed(713)
    torch.backends.cuda.matmul.allow_tf32 = False
    libraries = {'baseline': args.baseline, **candidates}
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  scope='Synthetic one-row native Q4 matrices. Identical weights/activations; exact production-output gate plus independent FP16 coefficient oracle. Warm CUDA graph ABBA timings include activation quantization and GEMV. No model or HTTP claim.',
                  libraries={name: dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                             for name, path in libraries.items()},
                  sources={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                           ('native/q4_residual.cu', 'scripts/experiments/q4_residual_launch.cu',
                            'dllm/native_q4_residual.py', __file__)}, checks=[], timings=[])
    def terminate(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, terminate)
    try:
        for outputs, width in ((113, 4096), (6144, 4096), (4096, 4096),
                               (28672, 4096), (4096, 14336), (32000, 4096)):
            weights, scales = quantize_q4(torch.randn(outputs, width, device='cuda', dtype=torch.float16) * .02)
            decoded = dequantize_q4(weights, scales, torch.float16)
            x = torch.randn(1, width, device='cuda', dtype=torch.float16)
            bias = torch.randn(outputs, device='cuda', dtype=torch.float16)
            for exponent in (-14, 0, 10):
                value = (x.float() * 2.**exponent).half()
                for biased in (False, True):
                    offset = bias if biased else None
                    expected = torch.nn.functional.linear(value, decoded, offset)
                    baseline = linear(value, weights, scales, offset, library_path=args.baseline)
                    for name, path in libraries.items():
                        actual = linear(value, weights, scales, offset, library_path=path)
                        error = actual.float() - expected.float()
                        rms = (error.square().mean() / expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        exact = torch.equal(actual, baseline)
                        passed = exact and bool(torch.isfinite(actual).all()) and rms < .001
                        report['checks'].append(dict(candidate=name, outputs=outputs, width=width,
                            activation_exponent=exponent, bias=biased, exact_baseline=exact,
                            relative_rms=rms, passed=passed))
                        assert passed, report['checks'][-1]
            for name, path in candidates.items():
                samples = {'baseline': [], name: []}
                for mode in ('baseline', name, name, 'baseline'):
                    samples[mode].append(triton.testing.do_bench_cudagraph(
                        lambda: linear(x, weights, scales, bias, library_path=libraries[mode]), rep=100))
                report['timings'].append(dict(candidate=name, outputs=outputs, width=width, samples_ms=samples,
                    baseline_ms=statistics.median(samples['baseline']), candidate_ms=statistics.median(samples[name])))
            print(outputs, width, 'passed', flush=True)
            del decoded, weights, scales, actual, baseline, expected
        report['passed'] = True
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
