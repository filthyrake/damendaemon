"""Experimental image/text fusion; not registered for serving.

The forward loop mirrors dllm's HybridExecutor while qualifying injected image
features and per-slot rotary offsets. Consolidate only after independent proof.
"""
import torch
from torch.nn import functional as F
from dllm.hybrid import HybridExecutor
from dllm.vision_positions import image_positions, rotary_tables


class ImageHybridExecutor(HybridExecutor):
    def __init__(self, model, *, max_batch=4, use_graph=False):
        self.pending=None
        self.rotary=None
        self.image_offsets=torch.zeros(max_batch,dtype=torch.long,device='cuda')
        super().__init__(model,use_graph=use_graph,max_batch=max_batch)

    def rotate(self,x,positions):
        if self.rotary is None:return super().rotate(x,positions)
        dim=self.c['rotary_dim']
        rotated,rest=x[...,:dim],x[...,dim:]
        left,right=rotated.chunk(2,-1)
        cos,sin=(table.unsqueeze(-2) for table in self.rotary)
        return torch.cat((rotated*cos+torch.cat((-right,left),-1)*sin,rest),-1).half()

    def prefill(self,tokens,slot=0):
        result=super().prefill(tokens,slot)
        self.image_offsets[slot]=0
        return result

    @torch.inference_mode()
    def prefill_images(self,tokens,features,grids,slot=0):
        if not tokens or len(tokens)>self.context or not 0<=slot<self.max_batch:
            raise ValueError('invalid image prompt or cache slot')
        positions,offset=image_positions(tokens,grids)
        if features.shape!=(tokens.count(248056),self.c['hidden_size']):
            raise ValueError('image feature shape does not match placeholders')
        if not bool(torch.isfinite(features).all()):raise ValueError('nonfinite image features')
        # Reference fusion casts features to the text embedding precision.
        tables=rotary_tables(positions.to(self.device))
        self.pending=(features.to(device=self.device,dtype=self.w['embed'].dtype),
                      tuple(table[None] for table in tables))
        self.lengths[slot]=0
        try:
            result=self.forward(torch.tensor([tokens],device=self.device),
                torch.arange(len(tokens),device=self.device)[None],
                torch.tensor([slot],device=self.device),True)
        finally:
            self.pending=None
            self.rotary=None
        self.image_offsets[slot]=offset
        self.lengths[slot]=len(tokens)
        self.length=self.lengths[0]
        return result

    @torch.inference_mode()
    def forward(self, ids, positions, slots, prefill):
        x = F.embedding(ids, self.w['embed']).float()
        if self.pending is not None:
            x[ids==248056]=self.pending[0].to(x.dtype)
            self.rotary=self.pending[1]
        elif prefill:
            self.rotary=None
        else:
            adjusted=positions+self.image_offsets[slots,None]
            tables=[rotary_tables(row.expand(3,-1)) for row in adjusted]
            self.rotary=tuple(torch.stack([table[i] for table in tables]) for i in (0,1))
        for i, kind in enumerate(self.c['layer_types']):
            h = self.norm(x, self.w[f'{i}.norm1'])
            x = x + (self.linear_attention(h, i, slots, prefill) if kind == 'linear_attention'
                     else self.attention(h, i, positions, slots, prefill))
            gate_up = self.linear(self.norm(x, self.w[f'{i}.norm2']), self.w[f'{i}.gate_up'])
            if self.fused:
                activated = self.dense_fused.swiglu(gate_up.reshape(-1, gate_up.shape[-1]), True)
                activated = activated.view(*gate_up.shape[:-1], -1)
            else:
                gate, up = gate_up.chunk(2, -1)
                activated = (F.silu(gate.float())*up.float()).half()
            x = x + self.linear(activated, self.w[f'{i}.down'])
        normalized = self.norm(x[:, -1], self.w['norm'])
        if self.w['head'].dtype == torch.int8:
            return self.linear(normalized.float(), self.w['head'])
        if self.fused:
            return self.fused.output_head(normalized, self.w['head'])
        return F.linear(normalized.float(), self.w['head'].float())
