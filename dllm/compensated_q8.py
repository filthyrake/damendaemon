"""Q8 projections with FP32 activations/output and three BF16 dot products.

High/high, high/low and low/high products retain more precision than a single
BF16 product. Omitting low/low makes this an approximation, not exact FP32 math.
Weights stay packed; only the current tile is expanded.
"""
import torch
import triton as tr
import triton.language as tl


def projection(x, weight, scales, bias=None):
    """Use compensated arithmetic for batches; preserve FP32 scalar matvec."""
    if x.dtype == torch.float32 and x.numel() // x.shape[-1] >= 2 and bias is None:
        return linear(x, weight, scales)
    return None


@tr.jit
def _project(X, W, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    groups = tl.arange(0, BK//32)
    high = tl.full((BM, BN), 0, tl.float32)
    low = tl.full((BM, BN), 0, tl.float32)
    for block in range(tr.cdiv(K, BK)):
        k = block*BK + inner
        x = tl.load(X + rows[:, None]*K + k[None, :], (rows[:, None]<M)&(k[None, :]<K), other=0)
        q = tl.load(W + cols[None, :]*K + k[:, None], (cols[None, :]<N)&(k[:, None]<K), other=0).to(tl.float32)
        g = block*(BK//32) + groups
        scale = tl.load(S + cols[None, :]*(K//32) + g[:, None],
                        (cols[None, :]<N)&(g[:, None]<K//32), other=0).to(tl.float32)
        w = (q.reshape(BK//32, 32, BN)*scale[:, None, :]).reshape(BK, BN)
        xh, wh = x.to(tl.bfloat16), w.to(tl.bfloat16)
        xl = (x-xh.to(tl.float32)).to(tl.bfloat16)
        wl = (w-wh.to(tl.float32)).to(tl.bfloat16)
        low = tl.dot(xh, wl, low)
        low = tl.dot(xl, wh, low)
        high = tl.dot(xh, wh, high)
    tl.store(Y + rows[:, None]*N + cols[None, :], high+low, (rows[:, None]<M)&(cols[None, :]<N))


def linear(x, weight, scales, *, row_tile=32, reduction_tile=32):
    if row_tile not in (16, 32, 64):
        raise ValueError('row tile must be 16, 32 or 64')
    if reduction_tile not in (32, 64, 128):
        raise ValueError('reduction tile must be 32, 64 or 128')
    if x.ndim < 1 or weight.ndim != 2 or scales.ndim != 2:
        raise ValueError('requires an input vector or batch and matrix weights/scales')
    if x.dtype != torch.float32 or weight.dtype != torch.int8 or scales.dtype != torch.float16 or x.shape[-1] % 32:
        raise ValueError('requires FP32 input and block-aligned Q8 weights')
    if x.shape[-1] == 0 or weight.shape[0] == 0 or weight.shape[1] != x.shape[-1] or scales.shape != (weight.shape[0], weight.shape[1]//32):
        raise ValueError('incompatible or empty Q8 projection dimensions')
    if x.device.type != 'cuda' or weight.device != x.device or scales.device != x.device:
        raise ValueError('Q8 projection tensors must share a CUDA/ROCm device')
    if not weight.is_contiguous() or not scales.is_contiguous():
        raise ValueError('Q8 weight and scale storage must be contiguous')
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    if x.shape[0] == 0:
        return out.reshape(*shape, weight.shape[0])
    _project[(tr.cdiv(x.shape[0], row_tile), tr.cdiv(weight.shape[0], 64))](
        x, weight, scales, out, x.shape[0], weight.shape[0], x.shape[1], row_tile, 64, reduction_tile,
        num_warps=4, num_stages=2)
    return out.reshape(*shape, weight.shape[0])
