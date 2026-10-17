"""Bounded explicit CUDA FP32 selected shared-KV attention with learned sink."""
import torch
from .selected_attention import _selected_attention


@torch.no_grad()
def selected_attention(query,kv,indices,sink,*,scale=None,query_chunk=16,key_chunk=64,
                       max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    if (not isinstance(query,torch.Tensor) or query.ndim!=4 or query.dtype!=torch.float32 or query.device.type!='cuda'
            or not 1<=query.shape[0]<=128 or not 0<=query.shape[1]<=2**20
            or not 1<=query.shape[2]<=4096 or not 1<=query.shape[3]<=4096):
        raise ValueError('CUDA selected attention requires bounded FP32 queries')
    b,t,h,d=query.shape
    if (not isinstance(kv,torch.Tensor) or kv.ndim!=3 or kv.shape[0]!=b or kv.shape[2]!=d
            or not 1<=kv.shape[1]<=2**20 or kv.dtype!=query.dtype or kv.device!=query.device
            or not isinstance(indices,torch.Tensor) or indices.ndim!=3 or indices.shape[:2]!=(b,t)
            or not 0<=indices.shape[2]<=2**20 or indices.dtype not in (torch.int32,torch.int64) or indices.device!=query.device
            or not isinstance(sink,torch.Tensor) or sink.shape!=(h,) or sink.dtype!=torch.float32 or sink.device!=query.device):
        raise ValueError('CUDA selected attention requires matching same-device KV/indices/sink')
    scale=d**-.5 if scale is None else scale
    if (type(scale) not in (int,float) or not 0<scale<=torch.finfo(torch.float32).max
            or type(query_chunk) is not int or not 1<=query_chunk<=128
            or type(key_chunk) is not int or not 1<=key_chunk<=256
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid CUDA selected attention options/budgets')
    q=min(t,query_chunk);k=min(indices.shape[2],key_chunk)
    output=4*b*t*h*d
    workspace=64*(b*t*h*d+indices.numel()+b*q*(h*k+k*d+h*d)+h)
    if output>max_output_bytes or workspace>max_workspace_bytes:
        raise ValueError('CUDA selected attention workspace/output budget exceeded')
    if (indices>=kv.shape[1]).any():raise ValueError('CUDA selected attention index outside cache')
    if not torch.isfinite(query).all() or not torch.isfinite(sink).all():raise ValueError('nonfinite CUDA query/sink')
    # Validate only gathered, nonpadding vectors inside each bounded chunk;
    # unselected/uninitialized cache slots remain irrelevant.
    return _selected_attention(query,kv,indices,sink,scale=scale,query_chunk=query_chunk,key_chunk=key_chunk,validate_finite=True)
