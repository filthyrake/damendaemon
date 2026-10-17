"""Bounded explicit CUDA FP32 learned DeepSeek compressed indexer."""
from functools import partial
import torch
from torch.nn import functional as F
from .floating_indexer import _index_hidden
from .cuda_prepared_projection import CudaPackedProjection
from .cuda_compressed_indexer import select_compressed


def _projection(value):
    return ((isinstance(value,torch.Tensor) and value.ndim==2) or isinstance(value,CudaPackedProjection)) and value.dtype==torch.float32 and value.device.type=='cuda'


def _linear(hidden,weight):
    return weight.linear(hidden) if isinstance(weight,CudaPackedProjection) else F.linear(hidden,weight)


@torch.no_grad()
def index_hidden(hidden,low_rank,query_weight,head_weight,keys,positions,inverse_frequencies,*,top_k,offset=0,
                 query_chunk=16,key_chunk=64,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or hidden.dtype!=torch.float32 or hidden.device.type!='cuda'
            or not 1<=hidden.shape[0]<=128 or not 0<=hidden.shape[1]<=2**20 or not 1<=hidden.shape[2]<=2**21
            or not isinstance(low_rank,torch.Tensor) or low_rank.ndim!=3 or low_rank.shape[:2]!=hidden.shape[:2]
            or not 1<=low_rank.shape[-1]<=2**21 or low_rank.dtype!=hidden.dtype or low_rank.device!=hidden.device
            or not isinstance(keys,torch.Tensor) or keys.ndim!=3 or keys.shape[0]!=hidden.shape[0]
            or not 0<=keys.shape[1]<=2**20 or not 32<=keys.shape[2]<=4096 or keys.shape[2]&(keys.shape[2]-1)
            or keys.dtype!=hidden.dtype or keys.device!=hidden.device):
        raise ValueError('CUDA learned indexer requires bounded same-device FP32 inputs')
    if any(not _projection(w) or w.device!=hidden.device for w in (query_weight,head_weight)):
        raise ValueError('CUDA indexer needs same-device dense or explicit packed projections')
    b,t,d=hidden.shape;n,w=keys.shape[1:];h=head_weight.shape[0];rank=low_rank.shape[-1]
    if not 1<=h<=4096 or head_weight.shape!=(h,d) or query_weight.shape!=(h*w,rank):
        raise ValueError('invalid CUDA learned indexer projection geometry')
    if (not isinstance(positions,torch.Tensor) or positions.shape!=(b,t) or positions.dtype!=torch.int64 or positions.device!=hidden.device
            or not isinstance(inverse_frequencies,torch.Tensor) or inverse_frequencies.ndim!=1
            or inverse_frequencies.dtype!=torch.float32 or inverse_frequencies.device!=hidden.device
            or not 1<=inverse_frequencies.numel()<=w//2):raise ValueError('invalid CUDA learned indexer positions/frequencies')
    if (type(top_k) is not int or not 1<=top_k<=4096 or type(offset) is not int or not 0<=offset<2**31
            or type(query_chunk) is not int or not 1<=query_chunk<=128 or type(key_chunk) is not int or not 1<=key_chunk<=256
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid CUDA learned indexer options/budgets')
    top=min(top_k,n);q=min(t,query_chunk);k=min(n,key_chunk)
    packed=max((v.options['workspace_bytes'] for v in (query_weight,head_weight) if isinstance(v,CudaPackedProjection)),default=0)
    dense=sum(v.numel() for v in (query_weight,head_weight) if isinstance(v,torch.Tensor))
    workspace=packed+256*(b*t*(d+rank)+b*n*w+b*q*(h*w+h*k+top+k)+dense)
    if workspace>max_workspace_bytes or 12*b*t*top>max_output_bytes:
        raise ValueError('CUDA learned indexer workspace/output budget exceeded')
    if (positions<0).any() or (positions>=2**20).any():raise ValueError('invalid CUDA learned indexer logical positions')
    selector=partial(select_compressed,max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
    return _index_hidden(hidden,low_rank,query_weight,head_weight,keys,positions,inverse_frequencies,
                         top_k=top_k,offset=offset,query_chunk=query_chunk,key_chunk=key_chunk,
                         projection_check=_projection,linear_fn=_linear,selector=selector,validate_finite=True)
