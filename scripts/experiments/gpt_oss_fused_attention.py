"""Experiment-only GPT-OSS executor for full-model sink-attention qualification."""
import torch
from torch.nn import functional as F
from dllm.gpt_oss import GptOssExecutor
from scripts.experiments.sink_attention import bounded_attention


class FusedAttentionExecutor(GptOssExecutor):
    def __init__(self,*args,ieee_attention=False,**kwargs):
        self.experimental_ieee_attention=ieee_attention
        self.experimental_prefill_slot=None
        super().__init__(*args,**kwargs)
        self.attention_projection_policy='experimental_ieee_attention_prefill_16_plus' if ieee_attention else 'experimental_compensated_attention'

    def forward(self,ids,positions,prefill,slot=0):
        self.experimental_prefill_slot=slot if prefill else None
        try:return super().forward(ids,positions,prefill,slot)
        finally:self.experimental_prefill_slot=None

    def layer(self,x,positions,slots,i,prefill=False,prefill_slot=None):
        if not prefill or x.shape[0]<16:return super().layer(x,positions,slots,i,prefill,prefill_slot=prefill_slot)
        # Direct layer replay has no host-side slot argument; that diagnostic
        # may synchronize. Normal prefill forwards the already-known integer.
        slot=prefill_slot if prefill_slot is not None else self.experimental_prefill_slot
        if slot is None:
            slot=int(slots[0])
            if not bool(torch.all(slots==slot)) or not torch.equal(positions,torch.arange(x.shape[0],device=x.device)):
                raise ValueError('experiment supports only zero-offset single-slot prefill')
        def project(value,weight,bias=None):
            if self.experimental_ieee_attention:
                return F.linear(value.float(),weight.float(),None if bias is None else bias.float())
            return self.dense_linear(value,weight,bias)
        w=self.w
        qkv=project(self.norm(x,w[f'{i}.norm1']),w[f'{i}.qkv.weight'],w[f'{i}.qkv.bias'])
        ck,cv=self.kv[i].unbind(0)
        query=self.kernels.rope_batch(qkv,self.cos,self.sin,positions,slots,ck,cv,self.heads,precise=True)
        window=self.c['sliding_window'] if self.c['layer_types'][i]=='sliding_attention' else 0
        attended=bounded_attention(query,ck[slot,:,:x.shape[0]],cv[slot,:,:x.shape[0]],w[f'{i}.sinks'],window)
        x=x+project(attended.reshape(x.shape[0],-1),w[f'{i}.out'],w[f'{i}.out.bias'])
        del qkv,query,attended
        return x+self.experts(self.norm(x,w[f'{i}.norm2']),i,prefill)
