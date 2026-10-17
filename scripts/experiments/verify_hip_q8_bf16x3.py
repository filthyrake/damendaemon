"""Real-GPU boundary checks for compensated Q8 and experimental Q4 projections.

No performance claim: the independent control expands signed bytes and stored
scales directly in FP32, then uses PyTorch linear with TF32 disabled.
"""
import argparse
import json
from pathlib import Path

import torch

from damendaemon.runtime import require_gpu
from damendaemon.compensated_q8 import linear as linear_q8, linear_q4


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantization', choices=('q8_0', 'q4_0'), default='q8_0')
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(319)
    torch.backends.cuda.matmul.allow_tf32 = False
    q4 = args.quantization == 'q4_0'
    linear = linear_q4 if q4 else linear_q8
    report = dict(passed=False, device=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), quantization=args.quantization, cases=[], rejected=[])
    try:
        # Cross tile edges in all dimensions; include the tiny decode boundary
        # even though the model wrapper dispatches only rows >=2.
        for m, n, k in ((1, 1, 32), (4, 63, 64), (5, 65, 96),
                        (31, 64, 32), (32, 65, 64), (33, 63, 96),
                        (127, 65, 5120), (128, 64, 96), (129, 63, 64)):
            for kind in ('normal', 'zero', 'strided'):
                x = torch.randn((m, k*2 if kind == 'strided' else k),
                                device='cuda', dtype=torch.float32)
                if kind == 'strided':
                    x = x[:, ::2]
                weight = (torch.randint(0, 256, (n, k//2), device='cuda', dtype=torch.uint8) if q4
                          else torch.randint(-128, 128, (n, k), device='cuda', dtype=torch.int8))
                scales = (torch.rand((n, k//32), device='cuda')*.001).half()
                if q4:
                    scales[:, ::2].neg_()
                if kind == 'zero':
                    scales.zero_()
                if q4:
                    # Independent upstream CPU GGUF expansion; no runtime
                    # nibble decoder or quantizer enters the Q4 oracle.
                    from gguf.quants import dequantize
                    from gguf.constants import GGMLQuantizationType
                    blocks = torch.cat((scales.cpu().contiguous().view(torch.uint8).reshape(-1, 2),
                                        weight.cpu().reshape(-1, 16)), dim=1).numpy()
                    expanded = torch.from_numpy(dequantize(blocks, GGMLQuantizationType.Q4_0)).reshape(n, k).cuda()
                else:
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
        weight = (torch.full((65, 32), 0x99, device='cuda', dtype=torch.uint8) if q4
                  else torch.ones((65, 64), device='cuda', dtype=torch.int8))
        scales = torch.ones((65, 2), device='cuda', dtype=torch.float16)
        # Empty batches and vectors must preserve ordinary linear shape rules.
        for value in (x[:0], x[0], x.reshape(1, 5, 64)):
            actual = linear(value, weight, scales)
            expected = torch.nn.functional.linear(value, torch.ones((65, 64), device='cuda'))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=.01)
        report['shape_checks'] = ['empty batch', 'vector', 'rank-three batch']
        invalid = {
            'input dtype': (x.half(), weight, scales),
            'weight dtype': (x, weight.float(), scales),
            'scale dtype': (x, weight, scales.float()),
            'input rank': (x[0, 0], weight, scales),
            'weight rank': (x, weight[0], scales),
            'scale rank': (x, weight, scales[0]),
            'width mismatch': (x, weight[:, :weight.shape[1]//2].contiguous(), scales[:, :1].contiguous()),
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
