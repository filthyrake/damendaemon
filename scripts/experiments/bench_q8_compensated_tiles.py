"""Matched-arithmetic ABBA tile probe for small batches, not serving performance."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from damendaemon.runtime import require_gpu
from scripts.experiments.hip_q8_bf16x3 import linear


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dimension', choices=('row', 'reduction'), default='row')
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(320)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, device=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), storage='q8_0',
                  arithmetic='three BF16 products, FP32 accumulation/input/output',
                  dimension=args.dimension, cases=[])
    tiles = (16, 32) if args.dimension == 'row' else (32, 64, 128)
    def project(x, weight, scales, tile):
        return linear(x, weight, scales, **{args.dimension+'_tile': tile})
    try:
        for m, n, k in ((2, 14336, 5120), (4, 48, 5120), (4, 14336, 5120),
                        (4, 34816, 5120), (4, 5120, 17408), (4, 248320, 5120)):
            x = torch.randn((m, k), device='cuda')
            weight = torch.randint(-128, 128, (n, k), device='cuda', dtype=torch.int8)
            scales = (torch.rand((n, k//32), device='cuda')*.001).half()
            expanded = (weight.float().reshape(n, k//32, 32)
                        * scales.float().unsqueeze(-1)).reshape(n, k)
            expected = torch.nn.functional.linear(x, expanded)
            del expanded
            checks = []
            for tile in tiles:
                actual = project(x, weight, scales, tile)
                error = actual-expected
                rms = (error.square().mean().sqrt()
                       / expected.square().mean().sqrt().clamp_min(1e-6)).item()
                maximum = error.abs().max().item()
                checks.append(dict(tile=tile, relative_rms=rms, max_absolute=maximum,
                                   passed=bool(torch.isfinite(actual).all()) and rms < 1e-4 and maximum < .01))
            case = dict(shape=[m, n, k], checks=checks, samples=[])
            report['cases'].append(case)
            if not all(check['passed'] for check in checks):
                raise AssertionError('tile numerical gate failed')
            for tile in tiles * 2:
                project(x, weight, scales, tile)
            torch.cuda.synchronize()
            for cycle in range(5):
                # Each candidate gets an ABBA comparison against tile 32.
                order = [tile for candidate in tiles if candidate != 32
                         for tile in (32, candidate, candidate, 32)]
                for tile in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = project(x, weight, scales, tile)
                    torch.cuda.synchronize()
                    case['samples'].append(dict(cycle=cycle, tile=tile, seconds=time.perf_counter()-start))
                    del result
            case['median_seconds'] = {str(tile): statistics.median(
                row['seconds'] for row in case['samples'] if row['tile'] == tile) for tile in tiles}
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
