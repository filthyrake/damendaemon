"""Explicit CUDA FP32 GLM packed FFN; no model dispatch or attention integration."""
import torch
from torch.nn import functional as F
from .cuda_packed_expert import CudaPackedExpert
from .expert_selection import select_experts
from .expert_routes import grouped_routes
from .glm_attention import _budgets


def _select(logits,bias,top_k,scale,device_routing=False):
    if device_routing:
        from .cuda_expert_routing import select_experts_device
        weights,ids=select_experts_device(logits,top_k,scoring='sigmoid',correction_bias=bias)
    else:
        weights,ids=select_experts(logits,top_k,scoring='sigmoid',correction_bias=bias,renormalize=False)
    return weights/(weights.sum(-1,keepdim=True)+1e-20)*scale,ids


def _routed(hidden,experts,selected,coefficients,shared):
    output=torch.zeros_like(hidden)
    for index,tokens,positions in grouped_routes(selected,len(experts)):
        for row in range(tokens.numel()):
            token=tokens[row:row+1]
            value=experts[index].forward(hidden.index_select(0,token))
            # GLM weights the completed down projection, never the activation.
            value=value*coefficients.reshape(-1).index_select(0,positions[row:row+1]).reshape(1,1)
            output.index_add_(0,token,value)
    output.add_(shared.forward(hidden))
    return output


class CudaPackedGlmFeedForward:
    """Borrow a resident GLM bank and router; compute one token at a time.

    Each expert supplies a bounded scratch allowance. Total workspace also
    reserves per-row routing/accumulation. Resident bank/router and final output
    are separate. Host count scheduling is explicit; this is not a fused kernel.
    """
    def __init__(self,experts,*,router=None,correction_bias=None,shared=None,top_k=8,
                 route_scale=2.5,activation_limit=None,device_routing=False,
                 max_output_bytes=64*1024**2,workspace_bytes=64*1024**2):
        dense=isinstance(experts,CudaPackedExpert)
        if dense:
            if any(v is not None for v in (router,correction_bias,shared)):
                raise ValueError('dense CUDA GLM FFN cannot include routing parameters')
            bank=(experts,)
        else:
            if not isinstance(experts,(tuple,list)) or not 2<=len(experts)<=4096:
                raise ValueError('CUDA GLM requires a prepared sparse expert sequence')
            bank=tuple(experts)
        all_experts=bank if dense else bank+(shared,)
        if any(not isinstance(e,CudaPackedExpert) for e in all_experts):
            raise ValueError('CUDA GLM requires prepared CUDA routed/shared experts')
        device,dim=bank[0].device,bank[0].width
        if device.type!='cuda' or any(e.device!=device or e.width!=dim or e.activation_limit!=activation_limit
                or any(m.format!='e4m3fn_f32' for m in (e.gate,e.up,e.down)) for e in all_experts):
            raise ValueError('CUDA GLM expert device/format/geometry/activation policy mismatch')
        count=0 if dense else len(bank)
        if not dense:
            if (not isinstance(router,torch.Tensor) or router.shape!=(count,dim)
                    or router.dtype!=torch.float32 or router.device!=device
                    or not isinstance(correction_bias,torch.Tensor) or correction_bias.shape!=(count,)
                    or correction_bias.dtype!=torch.float32 or correction_bias.device!=device
                    or type(top_k) is not int or not 1<=top_k<=count
                    or type(route_scale) not in (int,float) or not 0<route_scale<=torch.finfo(torch.float32).max):
                raise ValueError('invalid CUDA GLM routing geometry/policy')
        scratch=max(e.workspace_bytes for e in all_experts)+128*(dim+count)
        _budgets(0,scratch,max_output_bytes,workspace_bytes)
        if not dense and (not torch.isfinite(router).all() or not torch.isfinite(correction_bias).all()):
            raise ValueError('CUDA GLM routing parameters must be finite')
        self.device,self.dim,self.dense=device,dim,dense
        self.experts,self.shared=bank,shared
        self.router,self.correction_bias=router,correction_bias
        self.top_k,self.route_scale=top_k,route_scale
        if type(device_routing) is not bool:raise ValueError('CUDA GLM device routing must be boolean')
        self.device_routing=device_routing
        self.max_output_bytes,self.workspace_bytes=max_output_bytes,workspace_bytes
        self.required_workspace_bytes=scratch

    @torch.no_grad()
    def __call__(self,hidden):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or min(hidden.shape)<1
                or hidden.shape[-1]!=self.dim or hidden.dtype!=torch.float32 or hidden.device!=self.device):
            raise ValueError('CUDA GLM FFN needs same-device FP32 [batch,sequence,width]')
        _budgets(4*hidden.numel(),self.required_workspace_bytes,self.max_output_bytes,self.workspace_bytes)
        if not torch.isfinite(hidden).all():raise ValueError('CUDA GLM hidden states must be finite')
        output=torch.empty_like(hidden)
        for batch in range(hidden.shape[0]):
            for token in range(hidden.shape[1]):
                source=hidden[batch,token:token+1]
                if self.dense:value=self.experts[0].forward(source)
                else:
                    logits=F.linear(source,self.router)
                    if not torch.isfinite(logits).all():raise ValueError('CUDA GLM routing logits must be finite')
                    coefficients,ids=_select(logits,self.correction_bias,self.top_k,self.route_scale,self.device_routing)
                    value=_routed(source,self.experts,ids,coefficients,self.shared)
                if not torch.isfinite(value).all():raise ValueError('nonfinite CUDA GLM FFN output')
                output[batch,token:token+1]=value
        return output
