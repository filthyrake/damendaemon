"""Validation-only prompt tile override; production dispatch is unchanged."""
import hashlib
from pathlib import Path

import torch
import triton
from damendaemon import quantized_kernels_q4 as q4


def install_validation_dispatch():
    original = q4.linear

    def linear(x, weight, scales, bias=None):
        shape = x.shape[:-1]
        rows = x.numel() // x.shape[-1]
        width, outputs = x.shape[-1], weight.shape[0]
        if (not x.is_cuda or x.dtype != torch.float16 or rows not in (64, 129)
                or (width, outputs) not in ((4096, 6144), (4096, 4096),
                                           (4096, 28672), (14336, 4096),
                                           (4096, 32000))):
            return original(x, weight, scales, bias)
        x = x.reshape(rows, width).contiguous()
        result = torch.empty((rows, outputs), device=x.device, dtype=x.dtype)
        q4._gemm[(triton.cdiv(rows, 32), triton.cdiv(outputs, 64))](
            x, weight, scales, weight if bias is None else bias, result,
            rows, outputs, width, 32, 64, 128, bias is not None,
            num_warps=4, num_stages=3)
        return result.reshape(*shape, outputs)

    q4.linear = linear
    return dict(kind='experimental Q4 32-row prompt tiles', rows=[64, 129],
                scope='Mistral projection shapes; unchanged FP16 coefficients',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
