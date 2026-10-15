"""Explicit bounded CUDA DeepSeek packed routing and shared expert composition."""
import torch
from torch.nn import functional as F
from .cuda_packed_expert import CudaPackedExpert
from .cuda_packed_moe import cuda_packed_moe
from .expert_selection import select_experts
from .glm_attention import _budgets
from .reference_numerics import finite


class CudaPackedDeepSeekFeedForward:
    """Borrow immutable resident weights; bound routing scratch one token at a time.

    Unlike GLM, DeepSeek applies route weights to clipped expert activations
    before the down projection. Resident weights and allocator overhead remain
    separate from the scratch and output budgets. No automatic dispatch.
    """
    def __init__(self,experts,*,router,shared,correction_bias=None,token_table=None,
                 top_k=6,route_scale=1.5,activation_limit=10.,
                 max_output_bytes=64*1024**2,workspace_bytes=64*1024**2):
        if not isinstance(experts,(tuple,list)) or not 2<=len(experts)<=4096:
            raise ValueError('DeepSeek requires a bounded prepared CUDA expert bank')
        bank=tuple(experts);all_experts=bank+(shared,)
        if any(not isinstance(e,CudaPackedExpert) for e in all_experts):
            raise ValueError('DeepSeek requires prepared CUDA routed/shared experts')
        device,dim=bank[0].device,bank[0].width;count=len(bank)
        if (device.type!='cuda' or not 1<=dim<=2**21
                or type(activation_limit) not in (int,float) or not 0<activation_limit<=torch.finfo(torch.float32).max
                or any(e.device!=device or e.width!=dim or e.activation_limit!=activation_limit
                       or any(m.format not in ('e4m3fn','e2m1') for m in (e.gate,e.up,e.down)) for e in all_experts)):
            raise ValueError('DeepSeek CUDA expert placement/format/activation mismatch')
        if (not isinstance(router,torch.Tensor) or router.shape!=(count,dim)
                or router.device!=device or router.dtype!=torch.float32
                or type(top_k) is not int or not 1<=top_k<=count
                or type(route_scale) not in (int,float) or not 0<route_scale<=torch.finfo(torch.float32).max):
            raise ValueError('invalid DeepSeek CUDA router geometry/policy')
        if token_table is None:
            if (not isinstance(correction_bias,torch.Tensor) or correction_bias.shape!=(count,)
                    or correction_bias.device!=device or correction_bias.dtype!=torch.float32):
                raise ValueError('biased DeepSeek routing requires same-device FP32 correction bias')
        elif (correction_bias is not None or not isinstance(token_table,torch.Tensor) or token_table.ndim!=2
                or not 1<=token_table.shape[0]<=2**21 or token_table.shape[1]!=top_k
                or token_table.dtype not in (torch.int32,torch.int64) or token_table.device!=device):
            raise ValueError('hash DeepSeek routing requires only a bounded same-device integer table')
        # Preparation validation masks/sort are admitted before scanning values.
        table_scratch=0 if token_table is None else 32*token_table.numel()
        scratch=max(e.workspace_bytes for e in all_experts)+128*(dim+count+top_k)
        _budgets(0,max(scratch,8*router.numel()+table_scratch),max_output_bytes,workspace_bytes)
        finite(router,'DeepSeek router')
        if token_table is None:finite(correction_bias,'DeepSeek correction bias')
        elif ((token_table<0).any() or (token_table>=count).any()
              or (token_table.sort(-1).values.diff(dim=-1)==0).any()):
            raise ValueError('DeepSeek token table has invalid or duplicate routes')
        self.experts,self.shared,self.router=bank,shared,router
        self.correction_bias,self.token_table=correction_bias,token_table
        self.device,self.dim,self.top_k,self.route_scale=device,dim,top_k,route_scale
        self.activation_limit=activation_limit
        self.workspace_bytes,self.max_output_bytes=workspace_bytes,max_output_bytes
        self.required_workspace_bytes=scratch

    @torch.no_grad()
    def __call__(self,hidden,token_ids=None):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or not 1<=hidden.shape[0]<=128
                or not 1<=hidden.shape[1]<=2**20 or hidden.shape[2]!=self.dim
                or hidden.dtype!=torch.float32 or hidden.device!=self.device):
            raise ValueError('DeepSeek CUDA FFN requires bounded same-device FP32 hidden states')
        validation=hidden.numel()*4
        if self.token_table is not None:
            if (not isinstance(token_ids,torch.Tensor) or token_ids.shape!=hidden.shape[:2]
                    or token_ids.dtype not in (torch.int32,torch.int64) or token_ids.device!=self.device):
                raise ValueError('DeepSeek hash routing needs matching same-device token IDs')
            validation+=token_ids.numel()*16
        _budgets(hidden.numel()*4,max(self.required_workspace_bytes,validation),self.max_output_bytes,self.workspace_bytes)
        finite(hidden,'DeepSeek hidden states')
        if self.token_table is not None and ((token_ids<0).any() or (token_ids>=self.token_table.shape[0]).any()):
            raise ValueError('DeepSeek token ID outside table')
        output=torch.empty_like(hidden)
        for batch in range(hidden.shape[0]):
            for token in range(hidden.shape[1]):
                source=hidden[batch,token:token+1]
                logits=finite(F.linear(source,self.router),'DeepSeek routing logits')
                routes=None if self.token_table is None else self.token_table.index_select(0,token_ids[batch,token:token+1].long())
                coefficients,selected=select_experts(logits,self.top_k,scoring='sqrtsoftplus',renormalize=True,
                    scale=self.route_scale,correction_bias=self.correction_bias,selected_experts=routes)
                finite(coefficients,'DeepSeek route coefficients')
                value=cuda_packed_moe(source,self.experts,selected,coefficients,shared=self.shared)
                output[batch,token:token+1]=finite(value,'DeepSeek FFN output')
        return output
