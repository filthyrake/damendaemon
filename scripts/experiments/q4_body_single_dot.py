"""Validation-only contiguous Q4 body products; the output head is unchanged."""
import torch
import triton as tr
from scripts.experiments.q4_prefill import single_dot


def linear(x,weight,scales,bias=None):
    shape=x.shape[:-1];k=x.shape[-1];rows=x.numel()//k;n=weight.shape[0]
    x=x.reshape(rows,k).contiguous()
    output=torch.empty((rows,n),device=x.device,dtype=x.dtype)
    bm=max(16,tr.next_power_of_2(rows))
    single_dot[(tr.cdiv(rows,bm),tr.cdiv(n,64))](x,weight,scales,
        weight if bias is None else bias,output,rows,n,k,bm,64,128,bias is not None,
        num_warps=8 if bm==128 else 4,num_stages=3)
    return output.reshape(*shape,n)


def install_validation_dispatch():
    from dllm.quantization import Linear
    original_init,original_call=Linear.__init__,Linear.__call__
    def initialize(self,weights,*args,**kwargs):
        original_init(self,weights,*args,**kwargs)
        self._experiment_head_id=id(weights.get('head'))
    def call(self,x,weight,bias=None):
        if (x.is_cuda and x.dtype==torch.float16 and weight.dtype==torch.uint8
                and id(weight)!=self._experiment_head_id and 1<=x.numel()//x.shape[-1]<=128):
            return linear(x,weight,self.scales[id(weight)],bias)
        return original_call(self,x,weight,bias)
    Linear.__init__,Linear.__call__=initialize,call
