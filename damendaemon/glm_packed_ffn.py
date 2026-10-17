"""Prepared packed GLM experts, with FP32 CPU arithmetic and no dense banks."""
import torch
import torch.nn.functional as F
from .glm_attention import _float,_budgets
from .glm_ffn import GlmFeedForward,select_glm_experts
from .packed_moe import PackedExpert


class PackedGlmFeedForward(GlmFeedForward):
    """Borrow immutable experts; route one token at a time within a scratch cap.

    A single PackedExpert selects dense mode. A sequence plus floating router,
    correction bias and shared expert selects sparse mode. The workspace includes
    the largest expert allowance plus routing/accumulation tensors; resident
    weights, input, final output and allocator/BLAS overhead are excluded.
    """
    def __init__(self,experts,*,router=None,correction_bias=None,shared=None,top_k=8,route_scale=2.5,activation_limit=None,
                 max_output_bytes=64*1024**2,workspace_bytes=64*1024**2):
        if activation_limit is not None and (type(activation_limit) not in (int,float)
                or not 0<activation_limit<=torch.finfo(torch.float32).max):
            raise ValueError('invalid GLM packed FFN activation limit')
        self.activation_limit=activation_limit
        dense=isinstance(experts,PackedExpert)
        if dense:
            if any(v is not None for v in (router,correction_bias,shared)):
                raise ValueError('dense GLM packed FFN cannot include routing/shared parameters')
            bank=(experts,)
        else:
            if not isinstance(experts,(tuple,list)) or not 2<=len(experts)<=4096:
                raise ValueError('sparse GLM packed FFN needs a prepared expert sequence')
            bank=tuple(experts)
        all_experts=bank if dense else bank+(shared,)
        if any(not isinstance(expert,PackedExpert) for expert in all_experts):
            raise ValueError('GLM packed FFN requires prepared routed/shared experts')
        dim=bank[0].width
        if any(expert.width!=dim or expert.activation_limit!=activation_limit
               or any(matrix.format!='e4m3fn_f32' for matrix in (expert.gate,expert.up,expert.down))
               for expert in all_experts):
            raise ValueError('GLM packed expert geometry, format or activation policy mismatch')
        count=0 if dense else len(bank)
        if not dense:
            if (not _float(router,2) or router.shape!=(count,dim)
                    or not _float(correction_bias,1) or correction_bias.shape!=(count,)
                    or not torch.isfinite(router).all() or not torch.isfinite(correction_bias).all()
                    or type(top_k) is not int or not 1<=top_k<=count
                    or type(route_scale) not in (int,float) or not 0<route_scale<=torch.finfo(torch.float32).max):
                raise ValueError('invalid GLM packed expert routing geometry/policy')
        scratch=max(expert.workspace_bytes for expert in all_experts)+128*(dim+count)
        _budgets(0,scratch,max_output_bytes,workspace_bytes)
        self.kind='dense' if dense else 'sparse';self.dim=dim
        self.experts,self.shared=bank,shared
        self.router,self.correction_bias=router,correction_bias
        self.top_k,self.route_scale=top_k,route_scale
        self.max_output_bytes,self.workspace_bytes=max_output_bytes,workspace_bytes
        self.required_workspace_bytes=scratch

    @torch.no_grad()
    def __call__(self,hidden):
        if not _float(hidden,3) or hidden.shape[-1]!=self.dim or not torch.isfinite(hidden).all():
            raise ValueError('GLM packed FFN needs finite CPU FP32 hidden states')
        _budgets(4*hidden.numel(),self.required_workspace_bytes,self.max_output_bytes,self.workspace_bytes)
        output=torch.empty_like(hidden)
        for batch in range(hidden.shape[0]):
            for token in range(hidden.shape[1]):
                x=hidden[batch,token:token+1]
                if self.kind=='dense':value=self.experts[0].forward(x)
                else:
                    coefficients,ids=select_glm_experts(F.linear(x,self.router),self.correction_bias,
                                                       self.top_k,scale=self.route_scale)
                    value=torch.zeros_like(x)
                    for slot,index in enumerate(ids[0].tolist()):
                        # GLM applies coefficients AFTER down projection. Do not
                        # use PackedExpert's V4-style route_weights argument.
                        contribution=self.experts[index].forward(x)
                        value.add_(contribution*coefficients[0,slot])
                    value.add_(self.shared.forward(x))
                if not torch.isfinite(value).all():raise ValueError('nonfinite GLM packed FFN result')
                output[batch,token:token+1]=value
        return output
