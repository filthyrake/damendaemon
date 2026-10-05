"""Compensated FP32 prefill from packed K-quants using three FP16 products.

Weights are decoded directly into scaled high/low components. The low*low
term is omitted; this approximation is qualified against independent FP32.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .kquant_kernels import _values


@tr.jit
def _half_matmul(XH, XL, XS, WH, WL, WS, Y, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    block = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        k = start * BK + block
        xi = rows[:, None] * K + k[None, :]
        wi = cols[None, :] * K + k[:, None]
        xm = (rows[:, None] < M) & (k[None, :] < K)
        wm = (cols[None, :] < N) & (k[:, None] < K)
        xh = tl.load(XH + xi, xm, other=0)
        xl = tl.load(XL + xi, xm, other=0)
        wh = tl.load(WH + wi, wm, other=0)
        wl = tl.load(WL + wi, wm, other=0)
        # Three products retain both inputs' low terms; only low*low is omitted.
        acc += tl.dot(xh, wh) + tl.dot(xh, wl) + tl.dot(xl, wh)
    xs = tl.load(XS + rows, rows < M, other=1.)
    ws = tl.load(WS + cols, cols < N, other=1.)
    # Combine exponents before rescaling: acc*xs can overflow even when acc*xs*ws is finite.
    exponent = libdevice.ilogb(xs)[:, None] + libdevice.ilogb(ws)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], libdevice.ldexp(acc, exponent),
             (rows[:, None] < M) & (cols[None, :] < N))


@tr.jit
def _packed_split(W, HI, LO, S, K: tl.constexpr, SPEC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    values = _values(W, row + tl.arange(0, 1), k, K, SPEC).reshape(BLOCK)
    maximum = tl.max(tl.abs(values), 0)
    scale = tl.exp2(tl.maximum(-126., tl.minimum(120., tl.floor(tl.log2(maximum)) - 14.)))
    value = values / scale
    hi = value.to(tl.float16)
    lo = (value - hi.to(tl.float32)).to(tl.float16)
    tl.store(HI + row * K + k, hi, k < K)
    tl.store(LO + row * K + k, lo, k < K)
    tl.store(S + row, scale)


def linear(x, weight, layout, *, bm=32, bn=64):
    from dllm.precise_matmul import _split

    width, spec = layout
    if (x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous()
            or x.shape[1] != width or weight.dtype != torch.uint8 or weight.device != x.device):
        raise ValueError('expected contiguous FP32 CUDA inputs and packed weights')
    rows, outputs = x.shape[0], spec[-1][1]
    xh = torch.empty_like(x, dtype=torch.float16)
    xl = torch.empty_like(xh)
    xs = torch.empty(rows, device=x.device, dtype=torch.float32)
    wh = torch.empty((outputs, width), device=x.device, dtype=torch.float16)
    wl = torch.empty_like(wh)
    ws = torch.empty(outputs, device=x.device, dtype=torch.float32)
    _split[(rows,)](x, xh, xl, xs, width, tr.next_power_of_2(width))
    _packed_split[(outputs,)](weight, wh, wl, ws, width, spec, tr.next_power_of_2(width),
                              num_warps=8 if width >= 8192 else 4, enable_fp_fusion=False)
    result = torch.empty((rows, outputs), device=x.device, dtype=torch.float32)
    _half_matmul[(tr.cdiv(rows, bm), tr.cdiv(outputs, bn))](
        xh, xl, xs, wh, wl, ws, result, rows, outputs, width, bm, bn, 64,
        num_warps=4, num_stages=2)
    return result

