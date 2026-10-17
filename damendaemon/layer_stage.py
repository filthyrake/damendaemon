"""Contiguous decoder stage with device-local weights, KV and decode graphs.

The partition coordinator validates serving admission. GPU weights and KV belong only to this layer range;
only the first stage owns embeddings and only the last owns the output head.
"""
import copy
from itertools import accumulate
import torch
from damendaemon.batch import BatchExecutor


def shard_model(model,start,end):
    c=model.config
    if (c['model_type']!='qwen2' or model.quantization not in ('none','q8_0','q4_0')
            or getattr(model,'packed_formats',{}) or getattr(model,'awq_matrices',{})
            or c.get('rope_scaling') or not 0<=start<end<=c['num_hidden_layers']):
        raise ValueError('stage prototype requires FP16, Q8 or Q4 Qwen2 with standard RoPE')
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
    def __init__(self,model,start,end,max_batch=4,stage_graphs=False,kernels=None):
        from .partition_policy import q4_stage_adjacent_enabled,q4_stage_runtime_rope_enabled,validate
        adjacent=q4_stage_adjacent_enabled(model.quantization)
        runtime_rope=q4_stage_runtime_rope_enabled(model.quantization)
        self.runtime_rope_tokens=adjacent and runtime_rope
        if adjacent:
            validate(model,12,max_batch)
            if (start,end) not in ((0,12),(12,24)):raise ValueError('Q4 layout requires qualified stage ranges')
        self.q4_linear=None
        self.first=start==0
        self.last=end==model.config['num_hidden_layers']
        self.layer_range=(start,end)
        super().__init__(shard_model(model,start,end),use_graph=False,max_batch=max_batch,kernels=kernels)
        if adjacent:
            if self.device!='cuda' or self.compute_dtype!=torch.float16:
                raise ValueError('Q4 stage layout requires CUDA FP16')
            from .q4_stage import StageQ4Linear
            self.q4_linear=StageQ4Linear(self.quant_linear,self.w)
            self.linear=self.q4_linear
            if self.runtime_rope_tokens:
                from functools import partial
                self.rotary_store=partial(self.kernels.rope_store,runtime_tokens=True)
        self.stage_graphs={}
        if stage_graphs:
            self.capture_stage_graphs()

    def capture_stage_graphs(self):
        if self.device!='cuda':raise ValueError('stage graphs require CUDA')
        from damendaemon.cuda_warmup import warmup_stream
        for size in range(1,self.max_batch+1):
            shape=(size,) if self.first else (size,self.c['hidden_size'])
            inputs=torch.zeros(shape,device=self.device,
                dtype=torch.long if self.first else self.residual_dtype)
            positions=torch.zeros(size,device=self.device,dtype=torch.long)
            slots=torch.arange(size,device=self.device)
            with warmup_stream(),torch.inference_mode():
                for _ in range(3):self._decode(inputs,positions,slots)
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                # Torch's implicit capture stream can belong to another GPU
                # when a process owns multiple stages. Reuse this device's
                # serialized warmup stream, including its cuBLAS workspace.
                with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
                    output=self._decode(inputs,positions,slots)
            self.stage_graphs[size]=(inputs,positions,slots,graph,output)

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
    def run_prefill_packed(self,inputs,lengths,slots):
        if (not lengths or len(lengths)!=len(slots) or len(slots)>self.max_batch
                or any(type(length) is not int or not 1<=length<=self.context for length in lengths)
                or any(type(slot) is not int or not 0<=slot<self.max_batch for slot in slots)
                or len(set(slots))!=len(slots) or sum(lengths)>self.context
                or inputs.shape[0]!=sum(lengths)):
            raise ValueError('invalid packed stage prefill')
        positions=torch.tensor([position for length in lengths for position in range(length)],device=self.device)
        slot_ids=torch.tensor([slot for slot,length in zip(slots,lengths) for _ in range(length)],device=self.device)
        hidden=self.forward_packed_hidden(self.embedding(inputs) if self.first else inputs,
            positions,slot_ids,lengths,slots,[0]*len(slots))
        for slot,length in zip(slots,lengths):self.lengths[slot]=length
        if not self.last:return hidden
        ends=torch.tensor([end-1 for end in accumulate(lengths)],device=self.device)
        return self.output_logits(hidden.index_select(0,ends))

    @torch.inference_mode()
    def run_decode(self,inputs,positions,slots):
        if self.stage_graphs:
            cached_inputs,cached_positions,cached_slots,graph,output=self.stage_graphs[inputs.shape[0]]
            cached_inputs.copy_(inputs);cached_positions.copy_(positions);cached_slots.copy_(slots)
            graph.replay()
            return output
        return self._decode(inputs,positions,slots)

    def _decode(self,inputs,positions,slots):
        hidden=self.forward_batch_hidden(self.embedding(inputs) if self.first else inputs,positions,slots)
        return self.output_logits(hidden) if self.last else hidden
