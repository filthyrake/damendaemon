"""#68 compensated-prefill tile sweep on the frozen 27B projection shapes.

Extends diagnose_q4_prefill_projection.py: same synthetic packed matrices,
same FP64 reference from ordinary Torch Q4 coefficient decoding, same
numerical gates (max abs < 1e-4, rel RMS < 1e-4). Times each tile
configuration ABBA against the production baseline (32/32/4/2), reporting
seconds and weight-traffic relative speedup. A configuration only
qualifies if it passes numerics on every projection shape.

Requires ROCm, a 2 GiB allocator cap (DAMENDAEMON_ROCM_MEMORY_GIB=2), and
DAMENDAEMON_DEVICE=rocm, like the October control it extends.
"""
if not __debug__:
    raise RuntimeError('the tile sweep gates on assertions; run without -O')

import json
import os
import time

import torch

from damendaemon.compensated_q8 import linear_q4
from damendaemon.quantization import dequantize_q4
from damendaemon.rocm_policy import memory_limit
from damendaemon.runtime import require_gpu

# The three projection shapes of the pinned Qwen3.8-27B, frozen since October.
SHAPES = (('gate_up', 34816, 5120), ('down', 5120, 17408), ('qkv', 14336, 5120))
# (row_tile, reduction_tile, num_warps, num_stages); baseline first.
CONFIGS = ((32, 32, 4, 2), (64, 32, 4, 2), (64, 64, 4, 2), (64, 128, 4, 2),
           (128, 32, 4, 2), (128, 32, 8, 2), (128, 64, 4, 2), (128, 64, 8, 2),
           (128, 64, 8, 3), (128, 128, 8, 2), (128, 128, 8, 3))
ROUNDS = 3


def emit(**event):
    print(json.dumps(event), flush=True)


@torch.inference_mode()
def main():
    if os.environ.get('DAMENDAEMON_DEVICE') != 'rocm' or memory_limit() != 2 * 1024**3:
        raise ValueError('requires ROCm and a 2GiB allocator cap')
    require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(68)
    for name, outputs, width in SHAPES:
        weight = torch.randint(0, 256, (outputs, width // 2), device='cuda', dtype=torch.uint8)
        scales = torch.full((outputs, width // 32), .002, device='cuda', dtype=torch.float16)
        columns = torch.tensor([0, 1, outputs // 2, outputs - 2, outputs - 1], device='cuda')
        coefficients = dequantize_q4(weight[columns].cpu(), scales[columns].cpu()).double()
        rows = 127
        x = torch.randn(rows, width, device='cuda')
        expected = x.cpu().double() @ coefficients.T
        timing = {}
        for config in CONFIGS:
            row_tile, reduction_tile, warps, stages = config
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                value = linear_q4(x, weight, scales, row_tile=row_tile, reduction_tile=reduction_tile,
                                  num_warps=warps, num_stages=stages)
                torch.cuda.synchronize()
            except Exception as error:  # a tile that busts LDS is a finding, not a failure
                emit(event='compile_error', projection=name, config=config,
                     error_type=type(error).__name__, message=str(error)[:200])
                continue
            delta = value[:, columns].cpu().double() - expected
            absolute = delta.abs().max().item()
            rms = (delta.square().mean() / expected.square().mean()).sqrt().item()
            passed = absolute < 1e-4 and rms < 1e-4
            emit(event='numerical', projection=name, config=config, max_absolute=absolute,
                 relative_rms=rms, passed=passed,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if not passed:
                raise AssertionError(f'numerical gate failed for {name} at {config}')
            del value, delta
            for round_index in range(ROUNDS):
                for order in range(2):
                    first, second = CONFIGS[0], config
                    if order:  # ABBA so drift cannot favor one side
                        first, second = second, first
                    for member in (first, second):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        value = linear_q4(x, weight, scales, row_tile=member[0],
                                          reduction_tile=member[1], num_warps=member[2],
                                          num_stages=member[3])
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        del value
                        key = 'baseline' if member == CONFIGS[0] else str(member)
                        timing.setdefault(key, []).append(elapsed)
            best = min(timing[str(config)])
            base = min(timing['baseline'])
            emit(event='timing', projection=name, config=config, best_seconds=best,
                 baseline_best_seconds=base, speedup_vs_baseline=round(base / best, 3))
        del x, expected, weight, scales, columns, coefficients
        torch.cuda.empty_cache()
    emit(event='sweep_complete')


if __name__ == '__main__':
    main()
