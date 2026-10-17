"""Explicit CUDA DeepSeek block; CPU state ownership implementation is shared."""
from functools import partial
import torch
from .floating_block import FloatingBlock
from .cuda_floating_attention import _CudaBackend,attention_prefill
from .cuda_deepseek_ffn import CudaPackedDeepSeekFeedForward
from .cuda_hyper_connections import reduce_streams,restore_streams
from .cuda_packed_expert import CudaPackedExpert
from .reference_numerics import finite


class _Backend:
    expert_type=CudaPackedExpert

    def __init__(self,ffn,workspace_bytes,max_output_bytes):
        self.ffn,self.device=ffn,ffn.device
        self.workspace_bytes,self.max_output_bytes=workspace_bytes,max_output_bytes
        self.attention=_CudaBackend(workspace_bytes//2,workspace_bytes//2)
        bounds=dict(max_workspace_bytes=workspace_bytes//2,max_output_bytes=workspace_bytes//2)
        self.reduce=partial(reduce_streams,**bounds)
        self.restore=partial(restore_streams,**bounds)
        self.prefill=partial(attention_prefill,**bounds)

    def admit(self,hidden,token_ids,block):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=4 or hidden.dtype!=torch.float32
                or hidden.device!=self.device or not 1<=hidden.shape[0]<=128 or not 1<=hidden.shape[1]<=2**20
                or hidden.shape[2:]!=(block.hc_mult,block.dim)):
            raise ValueError('CUDA DeepSeek block requires matching bounded FP32 residual streams')
        b,t,h,d=hidden.shape
        # Reserve half for the inner operation and half for live residuals,
        # reductions, normalization, post weights and residual mixing matrices.
        overhead=128*b*t*(h*d+h*h+h+d)+8*sum(v.numel() for v in block.weights.values())
        if overhead>self.workspace_bytes//2 or 4*hidden.numel()>self.max_output_bytes:
            raise ValueError('CUDA DeepSeek block workspace/output budget exceeded')
        if block.table is not None:
            if (not isinstance(token_ids,torch.Tensor) or token_ids.shape!=(b,t)
                    or token_ids.device!=self.device or token_ids.dtype not in (torch.int32,torch.int64)):
                raise ValueError('CUDA DeepSeek block needs matching hash token IDs')
            if (token_ids<0).any() or (token_ids>=block.table.shape[0]).any():
                raise ValueError('CUDA DeepSeek token ID outside table')
        finite(hidden,'DeepSeek residual input')


class CudaDeepSeekBlock(FloatingBlock):
    def __init__(self,attention_weights,weights,frequencies,*,packed_experts,packed_shared,
                 groups,window,ratio=0,hc_mult=4,top_k=6,route_scale=1.5,index_top_k=512,
                 epsilon=1e-6,connection_epsilon=1e-6,iterations=20,activation_limit=10.,
                 device_routing=False,composition='per_token',
                 workspace_bytes=128*1024**2,max_output_bytes=64*1024**2):
        if (not isinstance(weights,dict) or type(workspace_bytes) is not int or not 2<=workspace_bytes<=2**31
                or type(max_output_bytes) is not int or not 1<=max_output_bytes<=2**31):
            raise ValueError('invalid CUDA DeepSeek block mapping/budgets')
        ffn=CudaPackedDeepSeekFeedForward(packed_experts,shared=packed_shared,router=weights.get('ffn.gate.weight'),
            correction_bias=weights.get('ffn.gate.bias'),token_table=weights.get('ffn.gate.tid2eid'),top_k=top_k,
            route_scale=route_scale,activation_limit=activation_limit,device_routing=device_routing,
            composition=composition,workspace_bytes=workspace_bytes//2,
            max_output_bytes=workspace_bytes//2)
        if (not isinstance(attention_weights,dict) or not _CudaBackend.projection(attention_weights.get('wq_a.weight'))
                or attention_weights['wq_a.weight'].shape[1]!=ffn.dim
                or attention_weights['wq_a.weight'].device!=ffn.device):
            raise ValueError('CUDA DeepSeek attention and expert geometry/placement disagree')
        backend=_Backend(ffn,workspace_bytes,max_output_bytes)
        super().__init__(attention_weights,weights,frequencies,groups=groups,window=window,ratio=ratio,
            hc_mult=hc_mult,top_k=top_k,route_scale=route_scale,index_top_k=index_top_k,epsilon=epsilon,
            connection_epsilon=connection_epsilon,iterations=iterations,activation_limit=activation_limit,
            workspace_bytes=workspace_bytes//2,packed_experts=packed_experts,packed_shared=packed_shared,_backend=backend)
