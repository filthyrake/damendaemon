"""Explicit bounded CUDA FP32 DeepSeek packed prefill and owned cached decode."""
from functools import partial
import torch
from torch.nn import functional as F
from . import floating_attention as core
from .cuda_prepared_projection import CudaPackedProjection
from .cuda_floating_indexer import index_hidden,_projection
from .cuda_floating_compressor import compress_hidden
from .cuda_selected_attention import selected_attention
from .reference_numerics import finite


class _CudaBackend:
    packed_type=CudaPackedProjection
    projection=staticmethod(_projection)

    def __init__(self,max_workspace_bytes,max_output_bytes):
        if any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes)):
            raise ValueError('invalid CUDA attention workspace/output budgets')
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes
        bounds=dict(max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_workspace_bytes)
        self.compress=partial(compress_hidden,**bounds)
        self.index=partial(index_hidden,**bounds)
        self.selected=partial(selected_attention,**bounds)

    @staticmethod
    def linear(hidden,weight):
        finite(hidden,'CUDA projection input')
        output=weight.linear(hidden) if isinstance(weight,CudaPackedProjection) else F.linear(hidden,weight)
        return finite(output,'CUDA projection output')

    def admit(self,hidden,weights,freq,*,groups,window,ratio,index_top_k,query_chunk,epsilon,capacity,max_cache_bytes):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or hidden.dtype!=torch.float32 or hidden.device.type!='cuda'
                or not 1<=hidden.shape[0]<=128 or not 1<=hidden.shape[1]<=2**20 or not 1<=hidden.shape[2]<=2**21
                or type(capacity) is not int or not hidden.shape[1]<=capacity<=2**20
                or type(window) is not int or not 1<=window<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31):
            raise ValueError('CUDA attention requires bounded FP32 input/cache geometry')
        heads,width,orank,index_width=core._check(hidden,weights,freq,groups,window,ratio,index_top_k,query_chunk,epsilon,self)
        rank=weights['q_norm.weight'].numel()
        if not 1<=heads<=4096 or not 1<=width<=4096 or not 1<=rank<=2**21 or not 1<=orank<=2**21:
            raise ValueError('CUDA attention projection geometry exceeds bounds')
        b,t,d=hidden.shape;blocks=capacity//ratio if ratio else 0;q=min(t,query_chunk);iw=index_width or 0
        cache=core.cache_bytes_per_slot(window,ratio,capacity,width,index_width,4)*b
        if cache>max_cache_bytes:raise ValueError('CUDA attention cache budget exceeded')
        packed=max((v.options['workspace_bytes'] for v in weights.values() if isinstance(v,CudaPackedProjection)),default=0)
        dense=sum(v.numel() for v in weights.values() if isinstance(v,torch.Tensor))
        factor=2 if ratio==4 else 1
        pool=b*ratio*factor*factor*(width+iw) if ratio else 0
        # Full raw prefill KV, combined cache, projections, pool replacement,
        # query/index/selected chunks and validation all precede publication.
        workspace=packed+256*(b*t*(d+rank+heads*width+groups*orank)+b*(window+blocks)*(width+iw)
                               +4*pool+dense+b*q*(heads*64+64*width+heads*width+4*iw+4*width))
        if workspace>self.max_workspace_bytes or 4*b*t*d>self.max_output_bytes:
            raise ValueError('CUDA attention workspace/output budget exceeded')
        finite(hidden,'CUDA attention hidden');finite(freq,'CUDA frequencies')
        for value in weights.values():
            if isinstance(value,torch.Tensor):finite(value,'CUDA attention parameter')

    def admit_reorder(self,state,indices):
        if indices.numel()>128 or 2*indices.numel()*state.bytes_per_slot>self.max_workspace_bytes:
            raise ValueError('CUDA attention reorder workspace/slot budget exceeded')
        if (indices<0).any() or (indices>=state.cache.shape[0]).any():
            raise ValueError('CUDA attention reorder index outside slots')


@torch.no_grad()
def attention_prefill(hidden,weights,inverse_frequencies,*,groups,window,ratio=0,index_top_k=512,query_chunk=16,
                      epsilon=1e-6,max_tokens=None,max_cache_bytes=256*1024**2,
                      max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    backend=_CudaBackend(max_workspace_bytes,max_output_bytes)
    capacity=hidden.shape[1] if max_tokens is None and isinstance(hidden,torch.Tensor) and hidden.ndim==3 else max_tokens
    backend.admit(hidden,weights,inverse_frequencies,groups=groups,window=window,ratio=ratio,index_top_k=index_top_k,
                  query_chunk=query_chunk,epsilon=epsilon,capacity=capacity,max_cache_bytes=max_cache_bytes)
    return core._attention_prefill(hidden,weights,inverse_frequencies,groups=groups,window=window,ratio=ratio,
                                   index_top_k=index_top_k,query_chunk=query_chunk,epsilon=epsilon,max_tokens=capacity,
                                   max_cache_bytes=max_cache_bytes,backend=backend)


@torch.no_grad()
def attention_decode(hidden,state,*,position):
    if not isinstance(state,core.AttentionState):raise ValueError('CUDA decode needs owned attention state')
    with state.lock:
        state._usable()
        if not isinstance(state.backend,_CudaBackend):raise ValueError('CUDA decode needs explicitly prepared CUDA state')
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or hidden.shape[1]!=1
                or hidden.shape[0]!=state.cache.shape[0] or hidden.device!=state.cache.device
                or type(position) is not int or position!=state.position or position>=state.capacity):
            raise ValueError('invalid CUDA decode input/cursor')
        state.backend.admit(hidden,state.weights,state.frequencies,groups=state.groups,window=state.window_size,ratio=state.ratio,
                            index_top_k=state.index_top_k,query_chunk=1,epsilon=state.epsilon,capacity=state.capacity,max_cache_bytes=state.max_cache_bytes)
        return core._attention_decode(hidden,state,position=position)
