"""Eager torch Q8_0 projections for Intel XPU bring-up.

Mirrors the CUDA Q8_0 boundary — dequantize to the activation dtype, then the
executor's dense linear — without any resident dequantized weight copy or
Triton dependency. Q4_0 remains rejected on XPU by policy.
"""
import torch
from torch.nn import functional as F

from .quantization import dequantize


@torch.inference_mode()
def linear(x, weight, scales, bias=None):
    shape = x.shape[:-1]
    matrix = dequantize(weight, scales, x.dtype)
    return F.linear(x.reshape(-1, x.shape[-1]), matrix, bias).reshape(*shape, matrix.shape[0])
