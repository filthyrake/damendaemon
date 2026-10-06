"""Device-local Q4 stage layout; host checkpoints remain canonical Q4_0.

Only qualified body projections use adjacent nibbles. The output head retains
its existing implementation. Repacking preserves allocations captured by graphs.
"""
import torch
import os


def runtime_rows_enabled():
    value=os.environ.get('DLLM_Q4_STAGE_RUNTIME_ROWS','1')
    if value not in ('0','1'):raise ValueError('DLLM_Q4_STAGE_RUNTIME_ROWS must be 0 or 1')
    return value=='1'


def use_runtime_rows(rows,width):
    # Partially filled 32/128-row down-projection tiles lose warm speed with
    # dynamic predicates. Retain their measured static implementation.
    return 1<=rows<=128 and (width==896 or (width==4864 and (rows<=16 or 32<rows<=64)))


@torch.inference_mode()
def repack_adjacent(weight):
    blocks=weight.reshape(weight.shape[0],-1,16)
    values=torch.cat((blocks&15,blocks>>4),dim=-1)
    return (values[...,::2]|(values[...,1::2]<<4)).reshape(weight.shape).contiguous()


class StageQ4Linear:
    def __init__(self,base,weights):
        self.base=base
        self.runtime_rows=runtime_rows_enabled()
        # Validate all projections before changing any representation. In
        # particular, a head/body alias must never silently repack the head.
        shapes={'qkv.weight':(1152,448),'out':(896,448),
                'gate_up':(9728,448),'down':(896,2432)}
        selected={}
        for name,weight in weights.items():
            prefix,_,suffix=name.partition('.')
            if prefix.isdigit() and suffix in shapes:
                if weight.dtype!=torch.uint8 or not weight.is_cuda or tuple(weight.shape)!=shapes[suffix]:
                    raise ValueError('unsupported Q4 stage body projection')
                if id(weight) not in base.scales:raise ValueError('missing Q4 stage scales')
                selected[id(weight)]=weight
        for name,weight in weights.items():
            if id(weight) in selected and not (name.partition('.')[0].isdigit() and name.partition('.')[2] in shapes):
                raise ValueError('Q4 stage body aliases another weight')
        if not selected:raise ValueError('empty Q4 stage body')
        self.weights=selected
        self.refresh()

    @torch.inference_mode()
    def refresh(self):
        """Call exactly once after uploading canonical checkpoint bytes."""
        for weight in self.weights.values():weight.copy_(repack_adjacent(weight))

    def __call__(self,x,weight,bias=None):
        if id(weight) not in self.weights:return self.base(x,weight,bias)
        from .q4_stage_kernels import single_dot_adjacent,single_dot_runtime_rows,expand_adjacent
        import triton as tr
        rows=x.numel()//x.shape[-1]
        scales=self.base.scales[id(weight)]
        if x.dtype!=torch.float16 or not 1<=rows<=128:
            return torch.nn.functional.linear(x,expand_adjacent(weight,scales,x.dtype),
                None if bias is None else bias.to(x.dtype))
        shape=x.shape[:-1];k=x.shape[-1];n=weight.shape[0]
        x=x.reshape(rows,k).contiguous()
        bm=max(16,tr.next_power_of_2(rows))
        bn,bk,stages=(64,64,3) if k==896 else ((32,128,1) if rows<=16 else (64,128,3))
        output=torch.empty((rows,n),device=x.device,dtype=x.dtype)
        kernel=single_dot_runtime_rows if self.runtime_rows and use_runtime_rows(rows,k) else single_dot_adjacent
        kernel[(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,weight,scales,
            weight if bias is None else bias,output,rows,n,k,bm,bn,bk,bias is not None,
            num_warps=8 if bm==128 else 4,num_stages=stages)
        return output.reshape(*shape,n)
