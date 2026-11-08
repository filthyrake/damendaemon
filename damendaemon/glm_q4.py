"""GLM blockwise-FP8 source decode and Q4_0 expert-bank packing (#719).

The official GLM-5.3 checkpoint ships blockwise-E4M3 FP8 weights (uint8
codes) with ``*_scale_inv`` companions on a 128x128 element grid including
partial tail blocks (e.g. kv_a 576 rows -> 5 scale rows). This module
decodes that source truth to FP32 and packs it into the repo's canonical
Q4_0 tile layout (uint8 paired nibbles + FP16 per-32 scales, format
``q4_0_f16``), producing prepared ``PackedMatrix``/``PackedExpert`` banks for
the qualified CUDA GLM executor path. Routers and combine gates are never
packed here: routing selection stays high precision (#668).

Decode and packing are measured against the real official checkpoint in
docs/evidence/glm/glm53-q4-fleet-719/ (rung 1: paired-delta drift 8.1-10.2%
rel RMS, byte-idempotent).
"""
import math

import torch

from .packed_moe import PackedExpert, PackedMatrix
from .quantization import quantize_q4

BLOCK = 128


def decode_blockwise_fp8(codes, scale_inv):
    """Decode one shipped blockwise-E4M3-FP8 tensor to its FP32 truth.

    ``codes`` is the raw uint8 storage; ``scale_inv`` the companion grid,
    ceil(rows/128) x ceil(columns/128), expanded over both axes (partial
    tail blocks included). Non-2D tensors and mismatched grids refuse.
    """
    if (not isinstance(codes, torch.Tensor) or not isinstance(scale_inv, torch.Tensor)
            or codes.ndim != 2 or scale_inv.ndim != 2):
        raise ValueError('blockwise FP8 decode requires 2-D codes and scale grid')
    rows, width = codes.shape
    grid = (-(-rows // BLOCK), -(-width // BLOCK))
    if tuple(scale_inv.shape) != grid:
        raise ValueError(f'blockwise FP8 scale grid {tuple(scale_inv.shape)} '
                         f'disagrees with {tuple(codes.shape)} (expected {grid})')
    if not hasattr(torch, 'float8_e4m3fn'):
        raise RuntimeError('PyTorch build without float8_e4m3fn support')
    row_index = torch.arange(rows, device=codes.device) // BLOCK
    column_index = torch.arange(width, device=codes.device) // BLOCK
    expanded = scale_inv[row_index][:, column_index].float()
    return codes.to(torch.float8_e4m3fn).float() * expanded


def q4_matrix(truth, **matrix_options):
    """Pack one FP32 truth matrix as a canonical Q4_0 ``PackedMatrix``."""
    if matrix_options:
        raise ValueError('PackedMatrix carries no options')
    if not isinstance(truth, torch.Tensor) or truth.ndim != 2 or truth.dtype != torch.float32:
        raise ValueError('Q4 packing requires a 2-D FP32 truth matrix')
    weight, scales = quantize_q4(truth)
    return PackedMatrix(weight, scales, 'q4_0_f16')


def q4_expert(truths, **expert_options):
    """Pack gate/up/down FP32 truth matrices as one Q4_0 ``PackedExpert``.

    ``truths`` is a (gate, up, down) triple of 2-D FP32 matrices.
    """
    if not isinstance(truths, tuple) or len(truths) != 3:
        raise ValueError('Q4 expert packing requires a (gate, up, down) truth triple')
    gate, up, down = (q4_matrix(t) for t in truths)
    return PackedExpert(gate, up, down, **expert_options)


def q4_bank_from_fp8(codes_and_scales, **expert_options):
    """Pack a bank of experts from shipped blockwise-FP8 source tensors.

    ``codes_and_scales`` is a sequence of (gate, up, down) triples, each
    element a (codes, scale_inv) pair. Returns the Q4_0 ``PackedExpert``
    tuple. The decode runs in bounded row slices so a full real bank never
    materializes every FP32 truth at once (256 experts x 3 matrices of a
    6144-wide GLM layer is ~12 GiB fp8 -> ~47 GiB fp32 if expanded whole).
    """
    bank = []
    for index, triple in enumerate(codes_and_scales):
        if not isinstance(triple, tuple) or len(triple) != 3:
            raise ValueError(f'bank entry {index} is not a (gate, up, down) triple')
        truths = []
        for pair in triple:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('bank matrices are (codes, scale_inv) pairs')
            codes, scale_inv = pair
            truths.append(_decode_rows(codes, scale_inv))
        bank.append(q4_expert(tuple(truths), **expert_options))
    return tuple(bank)


def _decode_rows(codes, scale_inv, *, slice_rows=64):
    """Decode in bounded row slices so FP32 truth peaks at slice size."""
    rows = codes.shape[0]
    out = torch.empty((rows, codes.shape[1]), dtype=torch.float32, device=codes.device)
    for start in range(0, rows, slice_rows):
        stop = min(rows, start + slice_rows)
        out[start:stop] = decode_blockwise_fp8(codes[start:stop],
                                               scale_inv[start // BLOCK:-(-stop // BLOCK)])
    return out


def projected_q4_bank_bytes(shapes):
    """Bytes one Q4 bank occupies: nibble payload plus FP16 scales per 32."""
    per_element = 0.5 + 2.0 / 32
    return int(math.ceil(sum(rows * width for rows, width in shapes) * per_element))
