"""Experimental image/text hybrid backend with opt-in serving dispatch.

Shares the text backbone with HybridExecutor and keeps image rotary offsets
separate from physical KV indices. Validation scope is documented separately.
"""
import torch
from torch.nn import functional as F
from .hybrid import HybridExecutor
from .vision_positions import image_positions, rotary_tables


class VisionHybridExecutor(HybridExecutor):
    requires_full_prefill=True
    def __init__(self, model, *, max_batch=4, use_graph=False):
        from .device import selected_device
        if (selected_device()!='cuda' or not torch.cuda.is_available()
                or torch.cuda.get_device_name()!='NVIDIA A2' or not 1<=max_batch<=4
                or model.max_context>1024):
            raise ValueError('experimental vision requires NVIDIA A2, context <=1024, and at most four slots')
        self.pending=None
        self.rotary=None
        self.image_offsets=torch.zeros(max_batch,dtype=torch.long,device='cuda')
        super().__init__(model,use_graph=use_graph,max_batch=max_batch)
        self.vision=None
        if getattr(model,'vision_weights',None) is not None:
            from .vision_qwen35 import VisionEncoder
            self.vision=VisionEncoder(model.vision_weights,device=self.device)

    def rotate(self,x,positions):
        if self.rotary is None:return super().rotate(x,positions)
        dim=self.c['rotary_dim']
        rotated,rest=x[...,:dim],x[...,dim:]
        left,right=rotated.chunk(2,-1)
        cos,sin=(table.unsqueeze(-2) for table in self.rotary)
        return torch.cat((rotated*cos+torch.cat((-right,left),-1)*sin,rest),-1).half()

    def prefill(self,tokens,slot=0):
        from .vision_model import ImagePrompt
        if isinstance(tokens,ImagePrompt):
            if not tokens or len(tokens)>self.context or type(slot) is not int or not 0<=slot<self.max_batch:
                raise ValueError('invalid image prompt or cache slot')
            if self.vision is None:raise ValueError('image weights are not loaded')
            features=self.vision.encode(tokens.pixels,tokens.grids)
            return self.prefill_images(tokens,features,tokens.grids,slot)
        result=super().prefill(tokens,slot)
        self.image_offsets[slot]=0
        return result

    def prefill_batch(self,prompts,slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(not p or len(p)>self.context for p in prompts)
                or any(type(slot) is not int or not 0<=slot<self.max_batch for slot in slots)):
            raise ValueError('invalid image/text prefill batch')
        return torch.cat([self.prefill(prompt,slot) for prompt,slot in zip(prompts,slots)])

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
        return self.forward_embeddings(x,positions,slots,prefill)
