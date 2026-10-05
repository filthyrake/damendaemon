"""Explicit contiguous decoder stage prototype, independent of transport.

Not registered for serving. GPU weights and KV belong only to this layer range;
only the first stage owns embeddings and only the last owns the output head.
"""
import copy
import torch
from dllm.batch import BatchExecutor


def shard_model(model,start,end):
    c=model.config
    if (c['model_type']!='qwen2' or model.quantization not in ('none','q8_0')
            or getattr(model,'packed_formats',{}) or getattr(model,'awq_matrices',{})
            or c.get('rope_scaling') or not 0<=start<end<=c['num_hidden_layers']):
        raise ValueError('stage prototype requires FP16 or Q8 Qwen2 with standard RoPE')
    shard=copy.copy(model)
    shard.config={**c,'num_hidden_layers':end-start}
    names={}
    for name in model.weights:
        prefix,_,tail=name.partition('.')
        if prefix.isdigit() and start<=int(prefix)<end:
            names[name]=f'{int(prefix)-start}.{tail}'
        elif (name=='embed' and start==0) or (name in ('norm','head') and end==c['num_hidden_layers']):
            names[name]=name
    shard.weights={renamed:model.weights[name] for name,renamed in names.items()}
    shard.scales={renamed:model.scales[name] for name,renamed in names.items() if name in model.scales}
    shard.bytes=sum(t.numel()*t.element_size() for t in
        {id(t):t for t in (*shard.weights.values(),*shard.scales.values())}.values())
    return shard


class LayerStage(BatchExecutor):
    def __init__(self,model,start,end,max_batch=4):
        self.first=start==0
        self.last=end==model.config['num_hidden_layers']
        self.layer_range=(start,end)
        super().__init__(shard_model(model,start,end),use_graph=False,max_batch=max_batch)

    def forward(self,ids,positions,prefill,slot=0):
        # Constructor warmup runs only this shard; no dummy global weight tensors.
        x=self.embedding(ids) if self.first else torch.zeros((ids.shape[0],self.c['hidden_size']),
            device=self.device,dtype=self.residual_dtype)
        hidden=self.forward_hidden(x,positions,prefill,slot)
        return self.output_logits(hidden[-1:]) if self.last else hidden

    @torch.inference_mode()
    def run_prefill(self,inputs,positions,slot):
        if not 0<=slot<self.max_batch or not 0<inputs.shape[0]<=self.context:
            raise ValueError('invalid stage prefill slot or length')
        hidden=self.forward_hidden(self.embedding(inputs) if self.first else inputs,positions,True,slot)
        self.lengths[slot]=inputs.shape[0]
        return self.output_logits(hidden[-1:]) if self.last else hidden

    @torch.inference_mode()
    def run_decode(self,inputs,positions,slots):
        hidden=self.forward_batch_hidden(self.embedding(inputs) if self.first else inputs,positions,slots)
        return self.output_logits(hidden) if self.last else hidden
