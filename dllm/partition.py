"""Opt-in same-host sequential layer partition behind the existing scheduler.

One scheduler thread owns both stages. Only hidden activations cross the stage
boundary; logits and sampling stay on the final GPU. A partial execution failure
poisons the whole executor until residency replaces it.
"""
import torch
from .prefill import groups
from .layer_stage import LayerStage,shard_model
from .partition_policy import validate,rebind_enabled,packed_prefill_enabled


def tensor_bytes(tensors):
    return sum(t.numel()*t.element_size() for t in {id(t):t for t in tensors}.values())


class PartitionExecutor:
    supports_chunked_prefill=False

    def __init__(self,model,split,use_graph=True,max_batch=4):
        validate(model,split,max_batch)
        rebind_enabled()
        self.packed_prefill=packed_prefill_enabled(model.quantization)
        if torch.cuda.device_count()!=2:raise ValueError('layer partition requires exactly two CUDA devices')
        self.model=model
        self.context=model.max_context
        self.max_batch=max_batch
        self.lengths=[0]*max_batch
        self.length=0
        self.device='cuda'
        self.output_device=1
        self.compute_dtype=self.residual_dtype=torch.float16
        self.kernel_backend='triton+layer-partition'+('+graphs' if use_graph else '')
        self.failed=False
        self.closed=False
        self.rebind_count=0
        self.stages=[]
        self.residency=[]
        try:
            for device,(start,end) in enumerate(((0,split),(split,model.config['num_hidden_layers']))):
                with torch.cuda.device(device):
                    stage=LayerStage(model,start,end,max_batch=max_batch,stage_graphs=use_graph,kernels='triton')
                    self.stages.append(stage)
                    scales=stage.quant_linear.scales.values() if stage.quant_linear else ()
                    self.residency.append(dict(device_index=device,layers=[start,end],
                        weight_layout='q4-adjacent-body' if stage.q4_linear is not None else 'canonical',
                        runtime_row_bounds=stage.q4_linear.runtime_rows if stage.q4_linear is not None else False,
                        runtime_rope_tokens=stage.runtime_rope_tokens,
                        weight_bytes=tensor_bytes(stage.w.values()),scale_bytes=tensor_bytes(scales),kv_bytes=tensor_bytes(stage.kv)))
        except BaseException:
            self.failed=True
            self.close()
            raise
        # The serialized owner also samples logits. Native sampling/grammar
        # kernels must launch on the final stage's device, without copying the
        # vocabulary logits back across PCIe every token.
        torch.cuda.set_device(self.output_device)

    def can_rebind(self,model):
        if (not rebind_enabled() or self.failed or self.closed or len(self.stages)!=2
                or model.config!=self.model.config or model.quantization!=self.model.quantization
                or model.max_context!=self.context):
            return False
        try:validate(model,self.stages[0].layer_range[1],self.max_batch)
        except ValueError:return False
        def aliases(values):
            first={}
            return {name:first.setdefault(id(value),name) for name,value in values.items()}
        for stage in self.stages:
            shard=shard_model(model,*stage.layer_range)
            if stage.w.keys()!=shard.weights.keys() or stage.model.scales.keys()!=shard.scales.keys():return False
            if aliases(stage.w)!=aliases(shard.weights):return False
            existing_scales={name:stage.quant_linear.scales[id(stage.w[name])] for name in shard.scales}
            if aliases(existing_scales)!=aliases(shard.scales):return False
            for name,target in stage.w.items():
                source=shard.weights[name]
                if target.shape!=source.shape or target.dtype!=source.dtype:return False
            for name,source in shard.scales.items():
                target=stage.quant_linear.scales[id(stage.w[name])]
                if target.shape!=source.shape or target.dtype!=source.dtype:return False
        return True

    @torch.inference_mode()
    def rebind(self,model):
        if not self.can_rebind(model):raise ValueError('incompatible partition weight rebind')
        try:
            for device in range(2):torch.cuda.synchronize(device)
            for device,stage in enumerate(self.stages):
                shard=shard_model(model,*stage.layer_range)
                with torch.cuda.device(device):
                    for name,target in stage.w.items():target.copy_(shard.weights[name])
                    for name,source in shard.scales.items():
                        stage.quant_linear.scales[id(stage.w[name])].copy_(source)
                    if stage.q4_linear is not None:stage.q4_linear.refresh()
                    torch.cuda.synchronize(device)
                stage.model=shard
                stage.c=shard.config
                stage.lengths=[0]*self.max_batch
            # Commit metadata only after every device has finished its copies.
            self.model=model
            self.lengths=[0]*self.max_batch
            self.length=0
            self.rebind_count+=1
            torch.cuda.set_device(self.output_device)
        except BaseException:
            self.failed=True
            self.lengths=[0]*self.max_batch
            raise

    def _ready(self):
        if self.failed or self.closed:raise RuntimeError('partition executor requires reactivation')

    def _slots(self,slots):
        if (not 1<=len(slots)<=self.max_batch or any(type(s) is not int or not 0<=s<self.max_batch for s in slots)
                or len(set(slots))!=len(slots)):
            raise ValueError('invalid partition slots')

    def _tokens(self,tokens):
        if not tokens or any(type(t) is not int or not 0<=t<self.model.config['vocab_size'] for t in tokens):
            raise ValueError('invalid partition token IDs')

    @torch.inference_mode()
    def prefill_batch(self,prompts,slots):
        self._ready();self._slots(slots)
        if len(prompts)!=len(slots):raise ValueError('prompt and slot counts differ')
        for prompt in prompts:
            self._tokens(prompt)
            if len(prompt)>self.context:raise ValueError('prompt exceeds partition context')
        outputs=[]
        try:
            batches=(groups(prompts,slots,self.context) if self.packed_prefill else
                     (([prompt],[slot]) for prompt,slot in zip(prompts,slots)))
            for batch,assigned in batches:
                lengths=[len(prompt) for prompt in batch]
                flat=[token for prompt in batch for token in prompt]
                hidden=None
                for device,stage in enumerate(self.stages):
                    with torch.cuda.device(device):
                        inputs=torch.tensor(flat,device='cuda') if device==0 else hidden.to(device)
                        hidden=(stage.run_prefill_packed(inputs,lengths,assigned) if len(batch)>1 else
                                stage.run_prefill(inputs,torch.arange(len(flat),device='cuda'),assigned[0]))
                outputs.append(hidden)
                for slot,length in zip(assigned,lengths):self.lengths[slot]=length
            torch.cuda.set_device(self.output_device)
            return torch.cat(outputs)
        except BaseException:
            self.failed=True
            self.lengths=[0]*self.max_batch
            raise

    @torch.inference_mode()
    def decode_batch(self,tokens,slots):
        self._ready();self._slots(slots);self._tokens(tokens)
        if len(tokens)!=len(slots) or any(not 0<self.lengths[s]<self.context for s in slots):
            raise ValueError('invalid partition decode position or batch')
        positions=[self.lengths[s] for s in slots]
        try:
            hidden=None
            for device,stage in enumerate(self.stages):
                with torch.cuda.device(device):
                    inputs=torch.tensor(tokens,device='cuda') if device==0 else hidden.to(device)
                    hidden=stage.run_decode(inputs,torch.tensor(positions,device='cuda'),torch.tensor(slots,device='cuda'))
            torch.cuda.set_device(self.output_device)
            next_tokens=hidden.argmax(-1)
            for slot in slots:self.lengths[slot]+=1
            return hidden,next_tokens
        except BaseException:
            self.failed=True
            self.lengths=[0]*self.max_batch
            raise

    def prefill(self,tokens,slot=0):
        result=self.prefill_batch([tokens],[slot]);self.length=len(tokens)
        return result

    def decode(self,token):
        result=self.decode_batch([token],[0]);self.length=self.lengths[0]
        return result

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            for device in range(2):torch.cuda.synchronize(device)
        finally:
            self.stages.clear()
            self.lengths=[0]*self.max_batch
