"""Explicit tiled CUDA packed projection foundation, not automatic model dispatch.

Raw weights/scales stay on one CUDA device. Only bounded tiles expand to FP32.
This uses PyTorch CUDA arithmetic, not hardware FP4/FP8 activation quantization,
and inherits the caller's FP32 matmul precision policy. No CPU route/value reads.
"""
import torch

from .packed_projection import projection_plan, validate_row_range
from .packed_tiles import _decode_tile


@torch.no_grad()
def cuda_packed_linear(hidden, weight, scales, *, format, row_chunk=64, column_chunk=256,
                       workspace_bytes=8*1024*1024, row_range=None):
    """CUDA [tokens,features] -> FP32, with the CPU control's scratch accounting.

    Resident tensors, final output, allocator overhead and BLAS workspace are
    excluded from the tile allowance. Callers must separately admit total VRAM.
    Weight, scale and hidden tensors may be strided but must share one device.
    Empty token batches allocate only an empty result and launch no decode.
    This is a capability foundation; it does not qualify expert routing/models.
    """
    rows,columns,token_chunk=projection_plan(weight,scales,format=format,row_chunk=row_chunk,
        column_chunk=column_chunk,workspace_bytes=workspace_bytes,device_type='cuda')
    first,last=validate_row_range(row_range,rows)
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=2 or hidden.device!=weight.device
            or hidden.dtype not in (torch.float16,torch.bfloat16,torch.float32) or hidden.shape[1]!=columns):
        raise ValueError('CUDA packed projection needs matching device and floating [tokens,features]')
    output=torch.zeros((hidden.shape[0],last-first),dtype=torch.float32,device=hidden.device)
    if hidden.shape[0]==0:return output
    for row in range(first,last,row_chunk):
        end_row=min(last,row+row_chunk)
        for column in range(0,columns,column_chunk):
            end_column=min(columns,column+column_chunk)
            tile=_decode_tile(weight,scales,format=format,row_start=row,row_stop=end_row,
                              column_start=column,column_stop=end_column)
            for token in range(0,hidden.shape[0],token_chunk):
                end_token=min(hidden.shape[0],token+token_chunk)
                inputs=hidden[token:end_token,column:end_column].float()
                output[token:end_token,row-first:end_row-first].add_(inputs@tile.T)
            del tile,inputs
    return output
