"""Explicit bounded prepared CUDA DeepSeek model; no checkpoint dispatch."""
import torch
from torch.nn import functional as F
from .floating_model import FloatingModel,ModelState
from .floating_attention import _norm
from .cuda_deepseek_block import CudaDeepSeekBlock
from .cuda_hyper_connections import reduce_head
from .reference_numerics import finite


class CudaDeepSeekModel(FloatingModel):
    def __init__(self,embedding,blocks,head,*,epsilon=1e-6,connection_epsilon=1e-6,
                 workspace_bytes=256*1024**2,max_output_bytes=64*1024**2):
        if (not isinstance(embedding,torch.Tensor) or embedding.ndim!=2 or embedding.device.type!='cuda'
                or embedding.dtype!=torch.float32 or not 1<=embedding.shape[0]<=2**21
                or not 1<=embedding.shape[1]<=2**21 or not isinstance(blocks,(list,tuple))
                or not 1<=len(blocks)<=1024 or any(not isinstance(b,CudaDeepSeekBlock) for b in blocks)
                or type(workspace_bytes) is not int or not 2<=workspace_bytes<=2**31
                or type(max_output_bytes) is not int or not 1<=max_output_bytes<=2**31):
            raise ValueError('CUDA DeepSeek model requires bounded FP32 embedding, CUDA blocks and budgets')
        if any(b.backend.workspace_bytes>workspace_bytes//2 for b in blocks):
            raise ValueError('CUDA DeepSeek block scratch exceeds model inner-operation allowance')
        super().__init__(embedding,blocks,head,epsilon=epsilon,connection_epsilon=connection_epsilon)
        self.workspace_bytes,self.max_output_bytes=workspace_bytes,max_output_bytes
        self.device=embedding.device

    def _admit(self,token_ids):
        if (not isinstance(token_ids,torch.Tensor) or token_ids.ndim!=2
                or not 1<=token_ids.shape[0]<=128 or not 1<=token_ids.shape[1]<=2**20
                or token_ids.device!=self.device or token_ids.dtype not in (torch.int32,torch.int64)):
            raise ValueError('CUDA DeepSeek tokens require bounded same-device integer rows')
        b,t=token_ids.shape;v,d=self.embedding.shape;h=self.blocks[0].hc_mult
        # Half the workspace remains available for one block/head operation.
        # Reserve live stream outputs, embedding indices and head validation.
        outer=32*b*t*(h*d+d+1)+8*sum(value.numel() for value in self.head.values())
        if outer>self.workspace_bytes//2 or 4*b*v>self.max_output_bytes:
            raise ValueError('CUDA DeepSeek model workspace/output budget exceeded')
        if (token_ids<0).any() or (token_ids>=v).any():
            raise ValueError('CUDA DeepSeek token outside vocabulary')

    def _embed(self,token_ids):
        self._admit(token_ids)
        return finite(super()._embed(token_ids),'DeepSeek embedded streams')

    def _logits(self,hidden):
        h=self.head
        reduced=reduce_head(hidden[:,-1:],h['hc_head_fn'],h['hc_head_scale'],h['hc_head_base'],
            norm_epsilon=self.epsilon,connection_epsilon=self.connection_epsilon,
            max_workspace_bytes=self.workspace_bytes//2,max_output_bytes=self.workspace_bytes//2)
        normalized=_norm(reduced,h['norm.weight'],self.epsilon)
        finite(h['head.weight'],'DeepSeek vocabulary projection')
        return finite(F.linear(normalized[:,0],h['head.weight']),'DeepSeek vocabulary logits')

    @torch.no_grad()
    def prefill(self,token_ids,*,max_tokens=None,max_cache_bytes=256*1024**2,query_chunk=16):
        self._admit(token_ids)
        capacity=token_ids.shape[1] if max_tokens is None else max_tokens
        if (type(capacity) is not int or not token_ids.shape[1]<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31
                or sum(b.cache_bytes_per_slot(capacity) for b in self.blocks)*token_ids.shape[0]>max_cache_bytes):
            raise ValueError('CUDA DeepSeek aggregate cache capacity/budget exceeded')
        return super().prefill(token_ids,max_tokens=capacity,max_cache_bytes=max_cache_bytes,query_chunk=query_chunk)

    @torch.no_grad()
    def decode(self,token_ids,state,*,position):
        if not isinstance(state,ModelState) or state.owner is not self:
            raise ValueError('CUDA DeepSeek state belongs to another or closed owner')
        with state.lock:
            state._usable()
            if (not isinstance(token_ids,torch.Tensor) or token_ids.shape!=(state.layers[0].attention.cache.shape[0],1)
                    or type(position) is not int or position!=state.position or position>=state.capacity):
                raise ValueError('invalid CUDA DeepSeek decode shape/cursor')
            self._admit(token_ids)
            return super().decode(token_ids,state,position=position)
