"""Isolated wider output tiles for the unchanged packed AWQ MMA kernel."""
import torch
import triton as tr
from damendaemon.awq_kernels import _gemm


def linear(x, matrix, bias=None, tile=128):
    if tile not in (64, 128, 256) or not 1 <= x.shape[0] <= 4:
        raise ValueError('AWQ tile experiment requires one to four rows')
    x = x.contiguous()
    rows, outputs = x.shape[0], matrix.outputs
    result = torch.empty((rows, outputs), device=x.device, dtype=torch.float16)
    _gemm[(1, tr.cdiv(outputs, tile))](x, matrix.weight, matrix.zeros, matrix.scales,
        matrix.weight if bias is None else bias, result, rows, matrix.inputs, outputs,
        matrix.group, 16, tile, 128, bias is not None,
        num_warps=8 if tile == 256 else 4,
        num_stages={64: 3, 128: 2, 256: 1}[tile])
    return result
