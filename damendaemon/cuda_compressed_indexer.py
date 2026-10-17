"""Explicit bounded CUDA FP32 DeepSeek compressed-key selection."""
import torch
from .compressed_indexer import _select_compressed


@torch.no_grad()
def select_compressed(query,keys,head_weights,positions,*,ratio,top_k,offset=0,query_chunk=16,key_chunk=64,
                      scale_head_weights=True,dot_scale=1.,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    if (not isinstance(query,torch.Tensor) or query.ndim!=4 or query.dtype!=torch.float32 or query.device.type!='cuda'
            or not 1<=query.shape[0]<=128 or not 0<=query.shape[1]<=2**20
            or not 1<=query.shape[2]<=4096 or not 1<=query.shape[3]<=4096):
        raise ValueError('CUDA indexer requires bounded FP32 query geometry')
    b,t,h,d=query.shape
    if (not isinstance(keys,torch.Tensor) or keys.ndim!=3 or keys.shape[0]!=b or keys.shape[2]!=d
            or not 0<=keys.shape[1]<=2**20 or keys.dtype!=query.dtype or keys.device!=query.device
            or not isinstance(head_weights,torch.Tensor) or head_weights.shape!=(b,t,h) or head_weights.dtype!=query.dtype or head_weights.device!=query.device
            or not isinstance(positions,torch.Tensor) or positions.shape!=(b,t) or positions.dtype!=torch.int64 or positions.device!=query.device):
        raise ValueError('CUDA indexer requires matching same-device inputs')
    if (type(ratio) is not int or not 1<=ratio<=128 or type(top_k) is not int or not 1<=top_k<=4096
            or type(offset) is not int or not 0<=offset<2**31
            or type(query_chunk) is not int or not 1<=query_chunk<=128
            or type(key_chunk) is not int or not 1<=key_chunk<=256
            or type(scale_head_weights) is not bool or type(dot_scale) not in (int,float)
            or not 0<dot_scale<=torch.finfo(torch.float32).max
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid CUDA indexer options/budgets')
    n=keys.shape[1];top=min(top_k,n);q=min(t,query_chunk);k=min(n,key_chunk)
    output=12*b*t*top
    workspace=64*(b*t*(h*d+h+1)+b*n*d+b*q*(h*k+2*(top+k)+h*d)+b*n)
    if output>max_output_bytes or workspace>max_workspace_bytes:
        raise ValueError('CUDA indexer workspace/output budget exceeded')
    if (positions<0).any() or (positions>=2**20).any():raise ValueError('invalid CUDA indexer logical positions')
    if not torch.isfinite(query).all() or not torch.isfinite(head_weights).all():raise ValueError('nonfinite CUDA indexer query/weights')
    visible=(positions.amax(1)+1)//ratio if t else torch.zeros(b,dtype=torch.int64,device=query.device)
    mask=torch.arange(n,device=query.device)[None,:]<visible[:,None]
    if (mask[:,:,None] & ~torch.isfinite(keys)).any():raise ValueError('nonfinite visible CUDA indexer keys')
    return _select_compressed(query,keys,head_weights,positions,ratio=ratio,top_k=top_k,offset=offset,
                              query_chunk=query_chunk,key_chunk=key_chunk,scale_head_weights=scale_head_weights,
                              dot_scale=dot_scale,validate_scores=True)
