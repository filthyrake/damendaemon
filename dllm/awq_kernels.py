"""Independent CUDA/Triton products over checkpoint-native AWQ GEMM packing.

Not yet connected to model loading. Coefficients round to FP16 before products;
projection accumulation uses FP32. No persistent expanded weight copy is kept.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _values(W, Z, S, columns, outputs, K: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr):
    mask = (columns < K) & (outputs < N)
    shift = ((outputs % 8)//2 + (outputs % 2)*4)*4
    packed = tl.load(W + columns*(N//8) + outputs//8, mask, other=0)
    zeros = tl.load(Z + (columns//GROUP)*(N//8) + outputs//8, mask, other=0)
    scales = tl.load(S + (columns//GROUP)*N + outputs, mask, other=0).to(tl.float32)
    integers = ((packed >> shift) & 15) - ((zeros >> shift) & 15)
    return (integers.to(tl.float32)*scales).to(tl.float16)


@tr.jit
def _expand(W, Z, S, Y, K: tl.constexpr, N: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    values = _values(W, Z, S, offsets//N, offsets%N, K, N, GROUP)
    tl.store(Y + offsets, values, offsets < K*N)


def expand(matrix):
    # Native input-major storage also avoids a second transpose/copy for GEMM.
    result = torch.empty((matrix.inputs, matrix.outputs), device=matrix.weight.device, dtype=torch.float16)
    _expand[(tr.cdiv(result.numel(), 1024),)](
        matrix.weight, matrix.zeros, matrix.scales, result,
        matrix.inputs, matrix.outputs, matrix.group, 1024)
    return result.T


@tr.jit
def _gemm(X, W, Z, S, B, Y, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
          GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BIASED: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    outputs = tl.program_id(1)*BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(tr.cdiv(K, BK)):
        columns = block*BK + offsets
        x = tl.load(X + rows[:, None]*K + columns[None, :],
                    (rows[:, None] < M) & (columns[None, :] < K), other=0)
        w = _values(W, Z, S, columns[:, None], outputs[None, :], K, N, GROUP)
        accumulator += tl.dot(x, w)
    if BIASED:
        accumulator += tl.load(B + outputs, outputs < N, other=0).to(tl.float32)[None, :]
    tl.store(Y + rows[:, None]*N + outputs[None, :], accumulator,
             (rows[:, None] < M) & (outputs[None, :] < N))


def linear(x, matrix, bias=None):
    if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.float16
            or x.device != matrix.weight.device or x.shape[1] != matrix.inputs or x.shape[0] < 1):
        raise ValueError('AWQ linear requires CUDA FP16 rows matching the packed matrix')
    if bias is not None and (bias.device != x.device or bias.dtype != x.dtype
                            or tuple(bias.shape) != (matrix.outputs,) or not bias.is_contiguous()):
        raise ValueError('invalid AWQ bias')
    x = x.contiguous()
    rows = x.shape[0]
    if rows > 128:
        return torch.nn.functional.linear(x, expand(matrix), bias)
    result = torch.empty((rows, matrix.outputs), device=x.device, dtype=x.dtype)
    bm = 16 if rows <= 16 else 32
    _gemm[(tr.cdiv(rows, bm), tr.cdiv(matrix.outputs, 64))](
        x, matrix.weight, matrix.zeros, matrix.scales,
        matrix.weight if bias is None else bias, result,
        rows, matrix.inputs, matrix.outputs, matrix.group, bm, 64, 128,
        bias is not None, num_warps=4, num_stages=3)
    return result
