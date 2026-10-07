"""Bounded CPU packed-weight projection control, not a GPU execution kernel."""
import torch

from .packed_tiles import MAX_TILE_ELEMENTS, decode_tile, validate_layout


def projection_plan(weight, scales, *, format, row_chunk, column_chunk, workspace_bytes):
    """Validate projection metadata and return rows, columns and token chunk."""
    rows, columns = validate_layout(weight, scales, format=format)
    if (any(type(v) is not int or v < 1 for v in (row_chunk, column_chunk, workspace_bytes))
            or row_chunk*column_chunk > MAX_TILE_ELEMENTS):
        raise ValueError('invalid packed projection tile dimensions or workspace allowance')
    r, c = min(row_chunk, rows), min(column_chunk, columns)
    weight_scratch = 128*(r*c+r+c)
    per_token = 8*c+8*r
    token_chunk = (workspace_bytes-weight_scratch)//per_token
    if token_chunk < 1:
        raise ValueError('packed projection workspace cannot hold one tile and token')
    return rows, columns, token_chunk


@torch.no_grad()
def packed_linear(hidden, weight, scales, *, format, row_chunk=64, column_chunk=256,
                  workspace_bytes=8*1024*1024, row_range=None):
    """Project CPU [tokens,features] to FP32 without a full weight expansion.

    E2M1/E4M3FN weights and E8M0 scales stay in raw uint8 storage; explicit
    e4m3fn_f32 instead retains FP32 block scales. Canonical q4_0_f16/q8_0_f16
    retain unsigned packed nibbles/signed bytes and FP16 row/32-column scales. Inputs may be
    strided FP16/BF16/FP32. Multiplication and accumulation use FP32; this does
    not emulate accelerator activation quantization or accumulation rounding.

    The scratch allowance conservatively reserves 128 bytes per decoded weight
    element and per tile axis entry, plus activation and product tiles. Resident
    inputs, final output, allocator overhead and BLAS scratch are excluded.
    Tile sizes are explicit; insufficient allowance fails before computation.
    """
    rows, columns, token_chunk = projection_plan(weight, scales, format=format, row_chunk=row_chunk,
                                                 column_chunk=column_chunk, workspace_bytes=workspace_bytes)
    first, last = validate_row_range(row_range, rows)
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
            or hidden.device.type != 'cpu' or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or hidden.shape[1] != columns):
        raise ValueError('packed projection needs CPU floating [tokens,features] matching the weight')
    output = torch.zeros((hidden.shape[0], last-first), dtype=torch.float32,device='cpu')
    if hidden.shape[0] == 0:
        return output
    for row in range(first, last, row_chunk):
        end_row = min(last, row+row_chunk)
        for column in range(0, columns, column_chunk):
            end_column = min(columns, column+column_chunk)
            tile = decode_tile(weight, scales, format=format, row_start=row, row_stop=end_row,
                               column_start=column, column_stop=end_column)
            for token in range(0, hidden.shape[0], token_chunk):
                end_token = min(hidden.shape[0], token+token_chunk)
                inputs = hidden[token:end_token, column:end_column].float()
                output[token:end_token, row-first:end_row-first].add_(inputs @ tile.T)
            # Drop before the next decode so two expanded weight tiles do not overlap.
            del tile, inputs
    return output


def validate_row_range(row_range, rows):
    if row_range is None:
        return 0, rows
    if (not isinstance(row_range,tuple) or len(row_range) != 2
            or any(type(v) is not int for v in row_range)
            or not 0 <= row_range[0] < row_range[1] <= rows):
        raise ValueError('projection row range must be a nonempty logical interval')
    return row_range
