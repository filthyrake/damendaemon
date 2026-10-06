"""Experimental adjacent-byte body storage; no production dispatch installs it.

Cached mode keeps canonical weights plus extra packed buffers. In-place mode
reuses existing buffers and expands adjacent data for large prefills. Both
refresh captured weight storage after canonical rebind or close on failure.
"""
import torch
import triton as tr
from scripts.experiments.q4_prefill import repack_adjacent,single_dot_adjacent,expand_adjacent


def install_validation_dispatch(gate_only=False,in_place=False):
    import os
    os.environ['DLLM_Q4_STAGE_ADJACENT']='0'  # Experiments own their representation.
    from dllm.quantization import Linear
    from dllm.partition import PartitionExecutor
    original_init,original_call=Linear.__init__,Linear.__call__
    original_rebind=PartitionExecutor.rebind
    shapes={(9728,896)} if gate_only else {(1152,896),(896,896),(9728,896),(896,4864)}
    def initialize(self,weights,*args,**kwargs):
        original_init(self,weights,*args,**kwargs)
        selected={id(weight):weight for name,weight in weights.items()
            if name!='head' and weight.is_cuda and weight.dtype==torch.uint8
            and (weight.shape[0],weight.shape[1]*2) in shapes}
        self.experimental_adjacent={}
        self.experimental_adjacent_inplace_ids=set(selected) if in_place else set()
        for key,weight in selected.items():
            packed=repack_adjacent(weight)
            if in_place:weight.copy_(packed)
            else:self.experimental_adjacent[key]=packed
    def call(self,x,weight,bias=None):
        stored_in_place=id(weight) in self.experimental_adjacent_inplace_ids
        cache=weight if stored_in_place else self.experimental_adjacent.get(id(weight))
        rows=x.numel()//x.shape[-1]
        if cache is None or not x.is_cuda or x.dtype!=torch.float16 or not 1<=rows<=128:
            if stored_in_place:
                return torch.nn.functional.linear(x,expand_adjacent(weight,self.scales[id(weight)],x.dtype),
                    None if bias is None else bias.to(x.dtype))
            return original_call(self,x,weight,bias)
        shape=x.shape[:-1];k=x.shape[-1];n=weight.shape[0];x=x.reshape(rows,k).contiguous()
        bm=max(16,tr.next_power_of_2(rows))
        bn,bk,stages=(64,64,3) if k==896 else ((32,128,1) if rows<=16 else (64,128,3))
        output=torch.empty((rows,n),device=x.device,dtype=x.dtype)
        single_dot_adjacent[(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,cache,self.scales[id(weight)],
            cache if bias is None else bias,output,rows,n,k,bm,bn,bk,bias is not None,
            num_warps=8 if bm==128 else 4,num_stages=stages)
        return output.reshape(*shape,n)
    def rebind(executor,model):
        result=original_rebind(executor,model)
        try:
            for device,stage in enumerate(executor.stages):
                with torch.cuda.device(device):
                    cached=getattr(stage.quant_linear,'experimental_adjacent',{})
                    inplace_ids=getattr(stage.quant_linear,'experimental_adjacent_inplace_ids',set())
                    for weight in {id(value):value for value in stage.w.values()}.values():
                        target=weight if id(weight) in inplace_ids else cached.get(id(weight))
                        if target is not None:target.copy_(repack_adjacent(weight))
                    torch.cuda.synchronize()
        except BaseException:
            executor.failed=True;executor.close();raise
        return result
    Linear.__init__,Linear.__call__=initialize,call
    PartitionExecutor.rebind=rebind
