"""Opt-in same-host sequential layer partition behind the existing scheduler.

One scheduler thread owns both stages. Only hidden activations cross the stage
boundary; logits and sampling stay on the final GPU. A partial execution failure
poisons the whole executor until residency replaces it.
"""
import torch
from .layer_stage import LayerStage
from .partition_policy import validate


def tensor_bytes(tensors):
    return sum(t.numel()*t.element_size() for t in {id(t):t for t in tensors}.values())


class PartitionExecutor:
    supports_chunked_prefill=False

    def __init__(self,model,split,use_graph=True,max_batch=4):
        validate(model,split,max_batch)
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
        self.stages=[]
        self.residency=[]
        try:
            for device,(start,end) in enumerate(((0,split),(split,model.config['num_hidden_layers']))):
                with torch.cuda.device(device):
                    stage=LayerStage(model,start,end,max_batch=max_batch,stage_graphs=use_graph)
                    self.stages.append(stage)
                    scales=stage.quant_linear.scales.values() if stage.quant_linear else ()
                    self.residency.append(dict(device_index=device,layers=[start,end],
                        weight_bytes=tensor_bytes(stage.w.values()),scale_bytes=tensor_bytes(scales),kv_bytes=tensor_bytes(stage.kv)))
        except BaseException:
            self.failed=True
            self.close()
            raise
        # The serialized owner also samples logits. Native sampling/grammar
        # kernels must launch on the final stage's device, without copying the
        # vocabulary logits back across PCIe every token.
        torch.cuda.set_device(self.output_device)

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
            for prompt,slot in zip(prompts,slots):
                hidden=None
                for device,stage in enumerate(self.stages):
                    with torch.cuda.device(device):
                        inputs=torch.tensor(prompt,device='cuda') if device==0 else hidden.to(device)
                        hidden=stage.run_prefill(inputs,torch.arange(len(prompt),device='cuda'),slot)
                outputs.append(hidden)
                self.lengths[slot]=len(prompt)
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
