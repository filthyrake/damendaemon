"""Isolated gfx1151 experiment: reuse Q8 weights across 2–4 matvec rows.

Not imported by the runtime. Activate before constructing an executor/graphs.
"""
import runpy
import sys
import torch
import triton as tr

from dllm import quantized_kernels as kernels
from dllm.device import selected_accelerator
from dllm.runtime import require_gpu

ordinary = kernels.linear


def candidate(x, weight, scales, bias=None):
    shape = x.shape[:-1]
    rows = x.numel() // x.shape[-1]
    if x.dtype != torch.float16 or not 2 <= rows <= 4:
        return ordinary(x, weight, scales, bias)
    x = x.reshape(rows, x.shape[-1]).contiguous()
    result = torch.empty((rows, weight.shape[0]), device=x.device, dtype=x.dtype)
    block = tr.next_power_of_2(x.shape[1])
    outputs = 2 if block >= 4096 else 8
    kernels._linear[(tr.cdiv(weight.shape[0], outputs),)](
        x, weight, scales, weight if bias is None else bias, result,
        x.shape[1], weight.shape[0], block, outputs, rows, bias is not None,
        num_warps=4, enable_fp_fusion=False)
    return result.reshape(*shape, weight.shape[0])


def activate():
    require_gpu()
    if selected_accelerator() != 'rocm' or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1151'):
        raise RuntimeError('experiment is restricted to the qualified HOMER target')
    kernels.linear = candidate


if __name__ == '__main__':
    activate()
    if len(sys.argv) < 2:
        raise SystemExit('supply the trusted local validation script to execute')
    if sys.argv[1:3] == ['--module', 'dllm']:
        del sys.argv[1:3]
        runpy.run_module('dllm', run_name='__main__')
    else:
        script = sys.argv.pop(1)
        sys.argv[0] = script
        runpy.run_path(script, run_name='__main__')
