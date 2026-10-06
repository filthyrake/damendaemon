"""Experimental extra packed cache; never installed by production dispatch.

Canonical Q4_0 weights remain resident for fallback and rebind. This prototype
therefore uses extra memory. Cached graph pointers survive explicit refresh.
"""
import torch
import triton as tr
from scripts.experiments.q4_prefill import repack_adjacent,single_dot_adjacent


def install_validation_dispatch(gate_only=False):
    from dllm.quantization import Linear
    from dllm.partition import PartitionExecutor
    original_init,original_call=Linear.__init__,Linear.__call__
    original_rebind=PartitionExecutor.rebind
    shapes={(9728,896)} if gate_only else {(1152,896),(896,896),(9728,896),(896,4864)}
    def initialize(self,weights,*args,**kwargs):
        original_init(self,weights,*args,**kwargs)
        self.experimental_adjacent={id(weight):repack_adjacent(weight) for name,weight in weights.items()
            if name!='head' and weight.is_cuda and weight.dtype==torch.uint8
            and (weight.shape[0],weight.shape[1]*2) in shapes}
    def call(self,x,weight,bias=None):
        cache=self.experimental_adjacent.get(id(weight))
        rows=x.numel()//x.shape[-1]
        if cache is None or not x.is_cuda or x.dtype!=torch.float16 or not 1<=rows<=128:
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
                    for weight in stage.w.values():
                        target=cached.get(id(weight))
                        if target is not None:target.copy_(repack_adjacent(weight))
                    torch.cuda.synchronize()
        except BaseException:
            executor.failed=True;executor.close();raise
        return result
    Linear.__init__,Linear.__call__=initialize,call
    PartitionExecutor.rebind=rebind
