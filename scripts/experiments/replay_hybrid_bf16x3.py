"""Bounded model replay: precise state/decode, compensated BF16 Q8 prefill."""
import runpy
import os
import torch
from scripts.experiments import replay_hybrid_fp32
from scripts.experiments.hip_q8_bf16x3 import linear as compensated
from dllm import quantized_kernels, hybrid

original_linear = quantized_kernels.linear
original_init = hybrid.HybridExecutor.__init__
minimum_rows = int(os.environ.get('DLLM_EXPERIMENT_BF16X3_MIN_ROWS', '5'))
if not 1 <= minimum_rows <= 128:
    raise ValueError('experimental compensated projection minimum rows must be 1..128')


def linear(x, weight, scales, bias=None):
    if x.dtype == torch.float32 and x.numel()//x.shape[-1] >= minimum_rows and bias is None:
        return compensated(x, weight, scales)
    return original_linear(x, weight, scales, bias)


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.kernel_backend += f'+diagnostic-bf16x3-min-rows-{minimum_rows}'


quantized_kernels.linear = linear
hybrid.HybridExecutor.__init__ = initialize
if __name__ == '__main__':
    runpy.run_path('scripts/replay_reference.py', run_name='__main__')
