"""Validation-only Q4 expansion with one packed-byte load for both nibbles."""
import torch
import triton
import triton.language as tl


@triton.jit
def _expand_pairs(W, S, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    packed_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = packed_offsets < TOTAL // 2
    packed = tl.load(W + packed_offsets, valid, other=0).to(tl.int32)
    scales = tl.load(S + packed_offsets // 16, valid, other=0).to(tl.float32)
    low = ((packed & 15) - 8).to(tl.float32) * scales
    high = ((packed >> 4) - 8).to(tl.float32) * scales
    offsets = (packed_offsets // 16) * 32 + packed_offsets % 16
    tl.store(Y + offsets, low, valid)
    tl.store(Y + offsets + 16, high, valid)


def expand(weight, scales, dtype, block=512):
    outputs, packed_width = weight.shape
    result = torch.empty((outputs, packed_width * 2), device=weight.device, dtype=dtype)
    _expand_pairs[(triton.cdiv(weight.numel(), block),)](
        weight, scales, result, result.numel(), block, num_warps=4)
    return result
