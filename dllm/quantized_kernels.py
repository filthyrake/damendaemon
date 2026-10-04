"""CUDA Q8_0 matrix-vector arithmetic; no resident dequantized weight copy."""
import torch
import triton as tr
import triton.language as tl
from .quantization import dequantize


@tr.jit
def _linear(X, W, S, BIAS, Y, K: tl.constexpr, N: tl.constexpr,
            BK: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, BIASED: tl.constexpr):
    outputs = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    quantized = tl.load(W + outputs[:,None]*K + columns[None,:],
        (outputs[:,None] < N) & (columns[None,:] < K), other=0).to(tl.float32)
    scale = tl.load(S + outputs[:,None]*(K//32) + columns[None,:]//32,
        (outputs[:,None] < N) & (columns[None,:] < K), other=0).to(tl.float32)
    # Match the executor's matrix precision after dequantization.
    weight = (quantized * scale).to(X.dtype.element_ty).to(tl.float32)
    for row in range(ROWS):
        x = tl.load(X + row*K + columns, columns < K, other=0).to(tl.float32)
        result = tl.sum(weight * x[None,:], 1)
        if BIASED:
            result += tl.load(BIAS + outputs, outputs < N, other=0).to(tl.float32)
        tl.store(Y + row*N + outputs, result, outputs < N)


def linear(x, weight, scales, bias=None):
    shape = x.shape[:-1]
    x = x.reshape(-1,x.shape[-1]).contiguous()
    if x.shape[0] > 4:
        result = torch.nn.functional.linear(x, dequantize(weight, scales, x.dtype), bias)
    else:
        result = torch.empty((x.shape[0],weight.shape[0]), device=x.device, dtype=x.dtype)
        block = tr.next_power_of_2(x.shape[1])
        outputs = 4 if block >= 4096 else 8
        _linear[(tr.cdiv(weight.shape[0],outputs),)](x,weight,scales,
            weight if bias is None else bias,result,x.shape[1],weight.shape[0],block,outputs,x.shape[0],bias is not None,
            num_warps=8 if block >= 4096 else 4, enable_fp_fusion=False)
    return result.reshape(*shape,weight.shape[0])
