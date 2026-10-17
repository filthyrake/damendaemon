"""Real-device numerical/ABBA probe of FP32 small-batch Q8 projection paths."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from damendaemon.runtime import require_gpu
from damendaemon.quantized_kernels import linear as current
from damendaemon.compensated_q8 import linear as compensated
from scripts.experiments.q8_blocked_matvec import linear as blocked


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(321)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, device=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  storage='q8_0', input_output_dtype='float32', cases=[],
                  arithmetic={'current': 'FP32 full-width reduction',
                              'blocked': 'FP32 tiled reduction',
                              'compensated': 'three BF16 products with FP32 accumulation'})
    operations = dict(current=current, compensated=compensated, blocked=blocked)
    try:
        for m, n, k in ((1, 1, 32), (2, 63, 96), (3, 65, 544),
                        (4, 48, 5120), (2, 14336, 5120), (4, 14336, 5120),
                        (4, 34816, 5120), (4, 5120, 17408), (4, 248320, 5120)):
            x = torch.randn((m, k), device='cuda')
            weight = torch.randint(-128, 128, (n, k), device='cuda', dtype=torch.int8)
            scales = (torch.rand((n, k//32), device='cuda')*.001).half()
            expanded = (weight.float().reshape(n, k//32, 32)
                        * scales.float().unsqueeze(-1)).reshape(n, k)
            expected = torch.nn.functional.linear(x, expanded)
            del expanded
            case = dict(shape=[m, n, k], checks=[], samples=[])
            report['cases'].append(case)
            for name, operation in operations.items():
                actual = operation(x, weight, scales)
                error = actual-expected
                rms = (error.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-6)).item()
                maximum = error.abs().max().item()
                case['checks'].append(dict(path=name, relative_rms=rms, max_absolute=maximum,
                    passed=bool(torch.isfinite(actual).all()) and rms < 1e-4 and maximum < .01))
            if not all(row['passed'] for row in case['checks']):
                raise AssertionError('primitive numerical gate failed')
            for _ in range(2):
                for operation in operations.values():
                    operation(x, weight, scales)
            torch.cuda.synchronize()
            for cycle in range(5):
                for candidate in ('compensated', 'blocked'):
                    for name in ('current', candidate, candidate, 'current'):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = operations[name](x, weight, scales)
                        torch.cuda.synchronize()
                        case['samples'].append(dict(cycle=cycle, pair=candidate, path=name,
                                                    seconds=time.perf_counter()-start))
                        del output
            case['median_seconds'] = {name: statistics.median(row['seconds'] for row in case['samples']
                                                              if row['path'] == name) for name in operations}
            del x, weight, scales, expected, actual, error
            torch.cuda.empty_cache()
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
