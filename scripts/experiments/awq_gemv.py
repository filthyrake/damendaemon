"""Isolated SIMT AWQ decode products with parallel input-group reduction."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(X, W, Z, S, P, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             BN: tl.constexpr):
    group = tl.program_id(1)
    columns = group*128 + tl.arange(0, 128)
    outputs = tl.program_id(0)*BN + tl.arange(0, BN)
    packed_outputs = tl.program_id(0)*(BN//8) + tl.arange(0, BN//8)
    lanes = tl.arange(0, 8)
    shifts = (lanes//2 + (lanes%2)*4)*4
    packed = tl.load(W + columns[:, None]*(N//8) + packed_outputs[None, :],
                     (columns[:, None] < K) & (packed_outputs[None, :] < N//8), other=0)
    zeros = tl.load(Z + group*(N//8) + packed_outputs, packed_outputs < N//8, other=0)
    scales = tl.load(S + group*N + outputs, outputs < N, other=0).to(tl.float32)
    values = ((packed[:, :, None] >> shifts[None, None, :]) & 15).reshape(128, BN)
    zero_values = ((zeros[:, None] >> shifts[None, :]) & 15).reshape(BN)
    weight = ((values-zero_values[None, :]).to(tl.float32)*scales[None, :]).to(tl.float16).to(tl.float32)
    for row in tl.static_range(M):
        x = tl.load(X + row*K + columns, columns < K, other=0).to(tl.float32)
        result = tl.sum(x[:, None]*weight, 0)
        tl.store(P + (group*M+row)*N + outputs, result, outputs < N)


@tr.jit
def _finish(P, B, Y, M: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr,
            BG: tl.constexpr, BIASED: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    groups = tl.arange(0, BG)
    values = tl.load(P + groups[:, None]*M*N + offsets[None, :],
                     (groups[:, None] < GROUPS) & (offsets[None, :] < M*N), other=0)
    result = tl.sum(values, 0)
    if BIASED:
        result += tl.load(B + offsets%N, offsets < M*N, other=0).to(tl.float32)
    tl.store(Y + offsets, result, offsets < M*N)


def linear(x, matrix, bias=None, tile=64):
    if tile not in (32, 64, 128, 256) or matrix.group != 128 or not 1 <= x.shape[0] <= 4:
        raise ValueError('GEMV experiment requires group-128 AWQ and one to four rows')
    x = x.contiguous()
    rows, outputs, groups = x.shape[0], matrix.outputs, matrix.inputs//128
    partial = torch.empty((groups, rows, outputs), device=x.device, dtype=torch.float32)
    result = torch.empty((rows, outputs), device=x.device, dtype=torch.float16)
    _partial[(tr.cdiv(outputs, tile), groups)](x, matrix.weight, matrix.zeros, matrix.scales,
        partial, rows, matrix.inputs, outputs, tile, num_warps=8 if tile == 256 else 4)
    _finish[(tr.cdiv(rows*outputs, 128),)](partial, partial if bias is None else bias,
        result, rows, outputs, groups, tr.next_power_of_2(groups), bias is not None, 128)
    return result
