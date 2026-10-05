"""Validation-only FP32 Tensor Core prefill; never rounds weights to FP16.

Compare three-term TF32 products with IEEE FP32 using the same decoded weights.
Both transient expansion and direct packed decode are profiled explicitly.
"""
import torch
import triton as tr
import triton.language as tl

from dllm.kquant_kernels import _values, expand


@tr.jit
def _matmul(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SPEC: tl.constexpr, PACKED: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    block = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        k = start * BK + block
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        if PACKED:
            w = tl.trans(_values(W, cols, k, K, SPEC))
        else:
            w = tl.load(W + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        acc += tl.dot(x, w, input_precision='tf32x3')
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def linear(x, weight, layout, *, packed=False, bm=32, bn=64, bk=32):
    width, spec = layout
    rows, outputs = x.shape[0], spec[-1][1]
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('expected contiguous CUDA FP32 input')
    dense = weight if packed else expand(weight, layout, torch.float32)
    result = torch.empty((rows, outputs), device=x.device, dtype=x.dtype)
    _matmul[(tr.cdiv(rows, bm), tr.cdiv(outputs, bn))](
        x, dense, result, rows, outputs, width, spec, packed, bm, bn, bk,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return result


from dllm.kquant_prefill_kernels import _half_matmul, linear as fused_linear


def compensated_linear(x, weight, layout, *, bm=32, bn=64, fused_split=False):
    if fused_split:
        return fused_linear(x, weight, layout, bm=bm, bn=bn)
    from dllm.precise_matmul import _split

    width, spec = layout
    rows, outputs = x.shape[0], spec[-1][1]
    xh = torch.empty_like(x, dtype=torch.float16)
    xl = torch.empty_like(xh)
    xs = torch.empty(rows, device=x.device, dtype=torch.float32)
    wh = torch.empty((outputs, width), device=x.device, dtype=torch.float16)
    wl = torch.empty_like(wh)
    ws = torch.empty(outputs, device=x.device, dtype=torch.float32)
    _split[(rows,)](x, xh, xl, xs, width, tr.next_power_of_2(width))
    dense = expand(weight, layout, torch.float32)
    _split[(outputs,)](dense, wh, wl, ws, width, tr.next_power_of_2(width))
    result = torch.empty((rows, outputs), device=x.device, dtype=torch.float32)
    _half_matmul[(tr.cdiv(rows, bm), tr.cdiv(outputs, bn))](
        xh, xl, xs, wh, wl, ws, result, rows, outputs, width, bm, bn, 64,
        num_warps=4, num_stages=2)
    return result


def install_validation_dispatch():
    import hashlib
    import inspect
    from pathlib import Path
    from dllm import kquant_kernels

    original = kquant_kernels.linear

    def experimental(x, weight, layout, bias=None):
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        width, spec = layout
        if (flat.is_cuda and flat.dtype == torch.float32 and flat.shape[0] >= 128
                and width in (2048, 8192) and all(kind in (12, 13, 14) for _, _, kind, _ in spec)):
            output = compensated_linear(flat, weight, layout, fused_split=True,
                                        bm=64 if flat.shape[0] >= 256 else 32, bn=64)
            if bias is not None:output += bias
            return output.reshape(*shape, spec[-1][1])
        return original(x, weight, layout, bias)

    kquant_kernels.linear = experimental
    return dict(name='compensated-fp32-prefill', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                kernel_sha256=hashlib.sha256(Path(inspect.getfile(fused_linear)).read_bytes()).hexdigest(),
                scope='FP32 K-quants, rows >=128, widths 2048/8192; high/low FP16 operands and three products',
                production_backend_label_excludes_experimental_override=True)
