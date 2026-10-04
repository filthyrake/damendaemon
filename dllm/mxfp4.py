"""MXFP4 block-layout validation; runtime projections stay packed on the GPU.

Format: 32 E2M1 values (low nibble first) plus one E8M0 scale per block.
See the pinned official GPT-OSS reference documented in issue #39.
"""
import torch


def validate(blocks,scales):
    if (blocks.dtype!=torch.uint8 or scales.dtype!=torch.uint8 or blocks.ndim<2
            or blocks.shape[-1]!=16 or scales.shape!=blocks.shape[:-1]
            or blocks.device!=scales.device or not blocks.is_contiguous() or not scales.is_contiguous()
            or any(size<=0 for size in blocks.shape)):
        raise ValueError('MXFP4 requires contiguous uint8 blocks[...,groups,16] and matching uint8 scales')
    return (*blocks.shape[:-2],blocks.shape[-2]*32)


def unpack(blocks,scales,dtype=torch.float32):
    """Native CUDA unpacking for diagnostics and bounded per-expert preparation."""
    shape=validate(blocks,scales)
    if blocks.device.type!='cuda' or dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('MXFP4 unpacking requires CUDA and FP32 or BF16 output')
    from .mxfp4_kernels import unpack as native
    return native(blocks,scales,shape,dtype)


def routed_linear(x,blocks,scales,selected,bias=None,shared_input=True):
    """Project only GPU-selected experts; never expand the complete model."""
    shape=validate(blocks,scales)
    if (len(shape)!=3 or x.device.type!='cuda' or x.ndim!=2 or x.dtype!=torch.float32
            or x.device!=blocks.device or not x.is_contiguous() or x.shape[1]!=shape[2]
            or selected.ndim!=2 or selected.dtype!=torch.int64 or selected.device!=x.device
            or not selected.is_contiguous() or not 1<=selected.shape[1]<=shape[0]
            or selected.shape[0]<1 or x.shape[0]!=(selected.shape[0] if shared_input else selected.numel())):
        raise ValueError('invalid MXFP4 routed projection')
    if bias is not None and (bias.shape!=shape[:2] or bias.dtype!=torch.float32
            or bias.device!=x.device or not bias.is_contiguous()):
        raise ValueError('MXFP4 expert bias must be contiguous FP32 [experts,outputs]')
    from .mxfp4_kernels import routed_linear as native
    return native(x,blocks,scales,selected,bias,shared_input)
