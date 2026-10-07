"""Bounded CPU FP32 dense and routed/shared GLM feed-forward equations."""
import torch
import torch.nn.functional as F
from .expert_selection import select_experts
from .glm_attention import _float,_budgets


def select_glm_experts(logits,bias,top_k,*,scale=2.5):
    if (not _float(logits,2) or not _float(bias,1) or bias.shape!=(logits.shape[1],)
            or not torch.isfinite(logits).all() or not torch.isfinite(bias).all()):
        raise ValueError('GLM routing requires finite CPU FP32 logits and correction bias')
    if type(scale) not in (int,float) or not 0<scale<=torch.finfo(torch.float32).max:
        raise ValueError('invalid GLM routing scale')
    weights,ids=select_experts(logits,top_k,scoring='sigmoid',correction_bias=bias,renormalize=False)
    # This additive epsilon matters for very small sigmoid scores; the shared
    # selector's underflow clamp is deliberately not used for GLM normalization.
    weights=weights/(weights.sum(-1,keepdim=True)+1e-20)*scale
    return weights,ids


class GlmFeedForward:
    def __init__(self,weights,*,kind='sparse',top_k=8,route_scale=2.5,
                 max_output_bytes=64*1024**2,workspace_bytes=64*1024**2):
        projections={'gate_proj.weight','up_proj.weight','down_proj.weight'}
        if kind=='dense':names=projections
        elif kind=='sparse':
            names={'gate.weight','gate.e_score_correction_bias'}|{'experts.'+n for n in projections}|{'shared_experts.'+n for n in projections}
        else:raise ValueError('GLM FFN must be dense or sparse')
        if not isinstance(weights,dict) or set(weights)!=names:raise ValueError('invalid GLM FFN parameters')
        for name,value in weights.items():
            rank=3 if name.startswith('experts.') else (1 if name.endswith('correction_bias') else 2)
            if not _float(value,rank) or not torch.isfinite(value).all():raise ValueError('GLM FFN weights must be finite CPU FP32')
        dim=weights['gate_proj.weight' if kind=='dense' else 'gate.weight'].shape[-1]
        def shapes(prefix,bank=False):
            gate=weights[prefix+'gate_proj.weight'];up=weights[prefix+'up_proj.weight'];down=weights[prefix+'down_proj.weight']
            width=gate.shape[-2]
            expected=(gate.shape[0],dim,width) if bank else (dim,width)
            if gate.shape[-1]!=dim or up.shape!=gate.shape or down.shape!=expected:
                raise ValueError('GLM FFN projection geometry mismatch')
            return width
        if kind=='dense':
            intermediate=shapes('');experts=0
        else:
            intermediate=max(shapes('experts.',True),shapes('shared_experts.'))
            experts=weights['gate.weight'].shape[0]
            if (experts<2 or weights['gate.e_score_correction_bias'].shape!=(experts,)
                    or weights['experts.gate_proj.weight'].shape[0]!=experts
                    or type(top_k) is not int or not 1<=top_k<=experts
                    or type(route_scale) not in (int,float) or not 0<route_scale<=torch.finfo(torch.float32).max):
                raise ValueError('GLM expert routing geometry/policy mismatch')
        _budgets(0,0,max_output_bytes,workspace_bytes)
        # Resident matrices are indexed by view, never gathered into route banks.
        per_row=4*(12*intermediate+8*dim+8*experts)+128
        rows=workspace_bytes//per_row
        if rows<1:raise ValueError('GLM FFN workspace cannot fit one token')
        self.weights=weights.copy();self.kind=kind;self.dim=dim
        self.top_k,self.route_scale=top_k,route_scale
        self.rows,self.max_output_bytes,self.workspace_bytes=rows,max_output_bytes,workspace_bytes

    @torch.no_grad()
    def __call__(self,hidden):
        if not _float(hidden,3) or hidden.shape[-1]!=self.dim:raise ValueError('invalid GLM FFN input')
        _budgets(4*hidden.numel(),0,self.max_output_bytes,self.workspace_bytes)
        if not torch.isfinite(hidden).all():raise ValueError('GLM FFN input must be finite')
        output=torch.empty_like(hidden)
        w=self.weights
        def expert(x,prefix,index=None):
            matrices=[w[prefix+name] for name in ('gate_proj.weight','up_proj.weight','down_proj.weight')]
            if index is not None:matrices=[matrix[index] for matrix in matrices]
            gate,up,down=matrices
            return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
        for batch in range(hidden.shape[0]):
            for start in range(0,hidden.shape[1],self.rows):
                stop=min(start+self.rows,hidden.shape[1]);x=hidden[batch,start:stop]
                if self.kind=='dense':value=expert(x,'')
                else:
                    coefficients,ids=select_glm_experts(F.linear(x,w['gate.weight']),w['gate.e_score_correction_bias'],
                                                       self.top_k,scale=self.route_scale)
                    value=torch.zeros_like(x)
                    for index in ids.unique(sorted=True).tolist():
                        tokens,routes=(ids==index).nonzero(as_tuple=True)
                        contribution=expert(x.index_select(0,tokens),'experts.',index)
                        # GLM weights after down projection, unlike V4's expert path.
                        contribution=contribution*coefficients[tokens,routes,None]
                        value.index_add_(0,tokens,contribution)
                    value=value+expert(x,'shared_experts.')
                if not torch.isfinite(value).all():raise ValueError('nonfinite GLM FFN result')
                output[batch,start:stop]=value
        return output
