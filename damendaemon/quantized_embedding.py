"""CUDA lookup directly from native Q4_0/Q8_0 rows; FP16 rounding is preserved."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _lookup(IDS, W, S, OUT, N: tl.constexpr, K: tl.constexpr,
            Q4: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(IDS + row)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (token >= 0) & (token < N) & (col < K)
    if Q4:
        packed = tl.load(W + token * (K // 2) + (col // 32) * 16 + col % 16,
                         valid, other=0).to(tl.int32)
        shift = tl.where(col % 32 < 16, 0, 4)
        value = ((packed >> shift) & 15) - 8
    else:
        value = tl.load(W + token * K + col, valid, other=0).to(tl.int32)
    scale = tl.load(S + token * (K // 32) + col // 32, valid, other=0).to(tl.float32)
    # The old CPU expansion rounded each product to FP16 before transfer or
    # widening to FP32. Keeping this boundary makes lookup bitwise equivalent.
    result = (value.to(tl.float32) * scale).to(tl.float16).to(OUT.dtype.element_ty)
    tl.store(OUT + row * K + col, result, col < K)


def embedding(ids, weight, scales, dtype):
    if (ids.device.type != 'cuda' or weight.device != ids.device or scales.device != ids.device
            or ids.dtype not in (torch.int32, torch.int64)
            or weight.dtype not in (torch.int8, torch.uint8) or scales.dtype != torch.float16
            or weight.ndim != 2 or not weight.is_contiguous() or not scales.is_contiguous()
            or dtype not in (torch.float16, torch.float32)):
        raise ValueError('invalid packed CUDA embedding layout')
    width = weight.shape[1] * (2 if weight.dtype == torch.uint8 else 1)
    if width % 32 or scales.shape != (weight.shape[0], width // 32):
        raise ValueError('invalid packed CUDA embedding scales')
    ids = ids.contiguous()
    result = torch.empty((*ids.shape, width), dtype=dtype, device=ids.device)
    if ids.numel():
        _lookup[(ids.numel(), tr.cdiv(width, 256))](ids, weight, scales, result,
            weight.shape[0], width, weight.dtype == torch.uint8, 256)
    return result
