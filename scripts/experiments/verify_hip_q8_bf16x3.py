"""Real-GPU boundary checks for experimental compensated Q8 projections.

No performance claim: the independent control expands signed bytes and stored
scales directly in FP32, then uses PyTorch linear with TF32 disabled.
"""
import argparse
import json
from pathlib import Path

import torch

from dllm.runtime import require_gpu
from scripts.experiments.hip_q8_bf16x3 import linear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(319)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, device=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), cases=[], rejected=[])
    try:
        # Cross tile edges in all dimensions; include the tiny decode boundary
        # even though the model wrapper currently dispatches only rows >4.
        for m, n, k in ((1, 1, 32), (4, 63, 64), (5, 65, 96),
                        (31, 64, 32), (32, 65, 64), (33, 63, 96),
                        (127, 65, 5120), (128, 64, 96), (129, 63, 64)):
            for kind in ('normal', 'zero', 'strided'):
                x = torch.randn((m, k*2 if kind == 'strided' else k),
                                device='cuda', dtype=torch.float32)
                if kind == 'strided':
                    x = x[:, ::2]
                weight = torch.randint(-128, 128, (n, k), device='cuda', dtype=torch.int8)
                scales = (torch.rand((n, k//32), device='cuda')*.001).half()
                if kind == 'zero':
                    scales.zero_()
                expanded = (weight.float().reshape(n, k//32, 32)
                            * scales.float().unsqueeze(-1)).reshape(n, k)
                expected = torch.nn.functional.linear(x, expanded)
                actual = linear(x, weight, scales)
                error = actual-expected
                rms = (error.square().mean().sqrt()
                       / expected.square().mean().sqrt().clamp_min(1e-6)).item()
                maximum = error.abs().max().item()
                passed = (actual.shape == expected.shape
                          and actual.dtype == torch.float32
                          and bool(torch.isfinite(actual).all())
                          and rms < 1e-4 and maximum < .01)
                report['cases'].append(dict(shape=[m, n, k], kind=kind,
                                           relative_rms=rms, max_absolute=maximum,
                                           passed=passed))
                if not passed:
                    raise AssertionError('numerical boundary gate failed')
        x = torch.randn((5, 64), device='cuda')
        weight = torch.ones((65, 64), device='cuda', dtype=torch.int8)
        scales = torch.ones((65, 2), device='cuda', dtype=torch.float16)
        # Empty batches and vectors must preserve ordinary linear shape rules.
        for value in (x[:0], x[0], x.reshape(1, 5, 64)):
            actual = linear(value, weight, scales)
            expected = torch.nn.functional.linear(value, weight.float())
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=.01)
        report['shape_checks'] = ['empty batch', 'vector', 'rank-three batch']
        invalid = {
            'input dtype': (x.half(), weight, scales),
            'weight dtype': (x, weight.float(), scales),
            'scale dtype': (x, weight, scales.float()),
            'input rank': (x[0, 0], weight, scales),
            'weight rank': (x, weight[0], scales),
            'scale rank': (x, weight, scales[0]),
            'width mismatch': (x, weight[:, :32].contiguous(), scales[:, :1].contiguous()),
            'scale shape': (x, weight, scales[:64]),
            'weight stride': (x, weight.t().contiguous().t(), scales),
            'scale stride': (x, weight, scales.t().contiguous().t()),
            'device mismatch': (x, weight.cpu(), scales),
            'empty width': (x[:, :0], weight[:, :0], scales[:, :0]),
        }
        for name, values in invalid.items():
            try:
                linear(*values)
            except ValueError:
                report['rejected'].append(name)
            else:
                raise AssertionError(f'accepted invalid {name}')
        torch.cuda.synchronize()
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
