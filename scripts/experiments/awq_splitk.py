"""Isolated AWQ split-K products; never enabled by ordinary model loading."""
import hashlib
from pathlib import Path
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(X, W, Z, S, P, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             SPLITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    split = tl.program_id(1)
    rows = tl.arange(0, BM)
    outputs = tl.program_id(0)*BN + tl.arange(0, BN)
    packed_outputs = tl.program_id(0)*(BN//8) + tl.arange(0, BN//8)
    lanes = tl.arange(0, 8)
    shifts = (lanes//2 + (lanes%2)*4)*4
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for block in range(split, tr.cdiv(K, 128), SPLITS):
        columns = block*128 + tl.arange(0, 128)
        x = tl.load(X + rows[:, None]*K + columns[None, :],
                    (rows[:, None] < M) & (columns[None, :] < K), other=0)
        packed = tl.load(W + columns[:, None]*(N//8) + packed_outputs[None, :],
                         (columns[:, None] < K) & (packed_outputs[None, :] < N//8), other=0)
        zeros = tl.load(Z + block*(N//8) + packed_outputs, packed_outputs < N//8, other=0)
        scales = tl.load(S + block*N + outputs, outputs < N, other=0).to(tl.float32)
        values = ((packed[:, :, None] >> shifts[None, None, :]) & 15).reshape(128, BN)
        zero_values = ((zeros[:, None] >> shifts[None, :]) & 15).reshape(BN)
        weight = ((values-zero_values[None, :]).to(tl.float32)*scales[None, :]).to(tl.float16)
        accumulator += tl.dot(x, weight)
    tl.store(P + (split*M+rows[:, None])*N + outputs[None, :], accumulator,
             (rows[:, None] < M) & (outputs[None, :] < N))


@tr.jit
def _finish(P, B, Y, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr,
            BIASED: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(P + splits[:, None]*M*N + offsets[None, :], offsets[None, :] < M*N, other=0)
    result = tl.sum(values, 0)
    if BIASED:
        result += tl.load(B + offsets%N, offsets < M*N, other=0).to(tl.float32)
    tl.store(Y + offsets, result, offsets < M*N)


def linear(x, matrix, bias=None, splits=4):
    if splits not in (2, 4, 8, 16) or matrix.group != 128 or not 1 <= x.shape[0] <= 4:
        raise ValueError('split-K experiment requires group-128 AWQ and one to four rows')
    x = x.contiguous()
    rows, outputs = x.shape[0], matrix.outputs
    partial = torch.empty((splits, rows, outputs), device=x.device, dtype=torch.float32)
    result = torch.empty((rows, outputs), device=x.device, dtype=torch.float16)
    _partial[(tr.cdiv(outputs, 64), splits)](x, matrix.weight, matrix.zeros, matrix.scales,
        partial, rows, matrix.inputs, outputs, splits, 16, 64, num_warps=4, num_stages=3)
    _finish[(tr.cdiv(rows*outputs, 256),)](partial, partial if bias is None else bias,
        result, rows, outputs, splits, bias is not None, 256)
    return result


def install_validation_dispatch(splits=4):
    from dllm import awq_kernels
    original = awq_kernels.linear
    awq_kernels.linear = lambda x, matrix, bias=None: (
        linear(x, matrix, bias, splits) if x.shape[0] <= 4 else original(x, matrix, bias))
    return dict(splits=splits, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
