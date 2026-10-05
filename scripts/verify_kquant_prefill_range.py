"""Probe compensated prefill scaling at extreme finite FP32 inputs."""
import argparse
import json
import hashlib
import inspect
from pathlib import Path

import torch

from dllm.gguf_model import GgufWeights
from dllm.kquant_kernels import expand
from experiments.kquant_prefill_f32 import compensated_linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GgufWeights(args.model, 512, 'none')
    source=Path(inspect.getfile(compensated_linear))
    report = dict(cases=[], passed=False, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  torch=torch.__version__, device=torch.cuda.get_device_name(), reference='FP64 product of identical decoded FP32 weights, rounded to FP32')
    try:
        with torch.inference_mode():
            for name in ('0.qkv.weight', '0.down'):
                weight = model.weights[name].cuda()
                layout = model.packed_formats[name]
                dense = expand(weight, layout, torch.float32)
                pattern = dense[0].sign().repeat(128, 1)
                for exponent in (-140, -100, 0, 100, 115):
                    x = pattern * (2. ** exponent)
                    expected = torch.nn.functional.linear(x.double(), dense.double()).float()
                    actual = compensated_linear(x, weight, layout, fused_split=True)
                    finite = bool(torch.isfinite(actual).all())
                    error = actual.double() - expected.double()
                    relative = (error.square().mean() / expected.double().square().mean().clamp_min(1e-300)).sqrt().item()
                    tolerance = expected.abs().max().double() * .001 + 2. ** -149
                    reference_finite = bool(torch.isfinite(expected).all())
                    passed = finite and reference_finite and bool((error.abs() <= tolerance).all())
                    report['cases'].append(dict(name=name, exponent=exponent, finite=finite, reference_finite=reference_finite,
                                                relative_rms=relative if finite else None, passed=passed))
                    print(report['cases'][-1], flush=True)
                del weight, dense
        report['passed'] = all(case['passed'] for case in report['cases'])
        if not report['passed']:raise AssertionError('compensated prefill range gate failed')
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
