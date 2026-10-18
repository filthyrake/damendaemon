"""Quantized KV cache for the hybrid executor: Q8_0 packed storage, dequantized attention.

Storage contract (mirrors the packed-weight convention in damendaemon/quantization.py):
the KV cache stores int8 values with per-32-channel FP16 scales along the channel
dimension. Values are dequantized to the compute dtype immediately before
attention; no second full-size dequantized cache stays resident. The per-layer
resident pair is (int8 data, FP16 scales), each shaped (2, batch, kv_heads,
context, dim[/32]) with axis 0 splitting K and V — the same outer layout as the
FP32 cache, so the executor's attention paths consume identical shapes.

Quantization granularity follows llama.cpp's K-cache convention: one FP16 scale
per 32 channels (or the largest divisor of the head dimension up to 32 for
small geometries). Zero scales (all-zero groups) store zeros, keeping padding
positions inert.
"""
import math

import torch

from .quantization import GROUP


def group_for(dim):
    """Channel-group size for a head dimension.

    Real model geometries (dim % 32 == 0) use llama.cpp's 32-channel K-cache
    convention. Small geometries (test fixtures, tiny head dims) fall back to
    the largest divisor of dim that is at most 32.
    """
    return math.gcd(int(dim), GROUP)


def allocate_kv(max_batch, kv_heads, context, dim, device):
    """Allocate the resident quantized KV pair for one layer.

    Returns (data, scales): data is (2, batch, kv_heads, context, dim) int8;
    scales is (2, batch, kv_heads, context, dim//group) FP16. Axis 0 is (K, V).
    """
    group = group_for(dim)
    if dim % group or context <= 0 or max_batch <= 0 or kv_heads <= 0:
        raise ValueError('invalid quantized KV geometry')
    data = torch.zeros((2, max_batch, kv_heads, context, dim), dtype=torch.int8, device=device)
    scales = torch.zeros((2, max_batch, kv_heads, context, dim // group),
                         dtype=torch.float16, device=device)
    return data, scales


def quantize_kv(values):
    """Q8_0-quantize KV values shaped (..., tokens, dim).

    Returns (int8 values, FP16 scales shaped (..., tokens, dim//group)). The
    scale is each channel group's absolute maximum divided by 127; zero groups
    store a zero scale.
    """
    dim = values.shape[-1]
    group = group_for(dim)
    if values.ndim < 2 or dim % group:
        raise ValueError('quantized KV requires a valid channel-group geometry')
    if not torch.isfinite(values).all():
        raise ValueError('cannot quantize non-finite KV values')
    groups = values.float().reshape(*values.shape[:-1], -1, group)
    scale = (groups.abs().amax(-1) / 127).to(torch.float16)
    inverse = torch.where(scale == 0, torch.zeros_like(scale), scale.float().reciprocal())
    normalized = (groups * inverse.unsqueeze(-1)).reshape(values.shape)
    rounded = (normalized.sign() * torch.floor(normalized.abs() + .5)).clamp(-127, 127)
    return rounded.to(torch.int8), scale


def dequantize_kv(data, scales, dtype):
    """Dequantize Q8_0 KV rows shaped (..., tokens, dim) to dtype."""
    if data.dtype != torch.int8 or scales.dtype != torch.float16:
        raise ValueError('quantized KV requires int8 values with FP16 scales')
    group = group_for(data.shape[-1])
    if data.shape[-1] % group or scales.shape != (*data.shape[:-1], data.shape[-1] // group):
        raise ValueError('incompatible quantized KV shapes')
    expanded = (data.float().reshape(*data.shape[:-1], -1, group)
                * scales.float().unsqueeze(-1)).reshape(data.shape)
    return expanded.to(dtype)


def store_kv(cache, slots, positions, k, v):
    """Quantize and write K/V at (slot, position) points of the resident pair.

    cache is the per-layer (data, scales) pair from allocate_kv. k/v are shaped
    (batch, kv_heads, tokens, dim); slots is a 1-D tensor of one slot per batch
    row; positions is a (batch, tokens) tensor of one absolute position per
    token per row (rows may hold different histories during batched decode).
    """
    data, scales = cache
    if positions.ndim != 2 or positions.shape[0] != k.shape[0] or positions.shape[1] != k.shape[2]:
        raise ValueError('quantized KV store requires (batch, tokens) positions')
    if (k.ndim != 4 or v.shape != k.shape or k.shape[0] != len(slots)
            or k.shape[-1] % group_for(k.shape[-1])):
        raise ValueError('quantized KV store requires matching batch-major K/V')
    if positions.numel() and int(positions.max()) >= data.shape[3]:
        raise ValueError('KV store position exceeds the cache context')
    packed_k, scale_k = quantize_kv(k)
    packed_v, scale_v = quantize_kv(v)
    # k: (batch, heads, tokens, dim); cache rows are (heads, context, dim).
    # packed rows are (heads, tokens, dim) — token-aligned with the row's positions.
    for row in range(k.shape[0]):
        slot = int(slots[row])
        row_positions = positions[row]
        data[0, slot][:, row_positions, :] = packed_k[row]
        data[1, slot][:, row_positions, :] = packed_v[row]
        scales[0, slot][:, row_positions, :] = scale_k[row]
        scales[1, slot][:, row_positions, :] = scale_v[row]


def read_kv(cache, slots, end):
    """Dequantize the populated prefix [0:end] for the selected slots.

    Returns (k, v) FP32 shaped (slots, kv_heads, end, dim) — slot-indexed,
    matching the FP32 cache's outer slot dimension so the executor's attention
    paths keep their existing shape contracts. Duplicate slots duplicate rows.
    """
    data, scales = cache
    selected_data = data.index_select(1, slots)[..., :end, :]
    selected_scales = scales.index_select(1, slots)[..., :end, :]
    k = dequantize_kv(selected_data[0], selected_scales[0], torch.float32)
    v = dequantize_kv(selected_data[1], selected_scales[1], torch.float32)
    return k, v
