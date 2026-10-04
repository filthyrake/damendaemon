"""Compare fused Q8 prompt tiles against expanded-weight cuBLAS on A2."""
import json
import torch
import triton
from dllm.quantized_kernels import _matmul, expand

for m,n,k in ((23,28672,4096),(46,28672,4096),(92,28672,4096),(92,4096,14336)):
    x=torch.randn(m,k,device='cuda',dtype=torch.float16)
    w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
    s=torch.full((n,k//32),.001,device='cuda',dtype=torch.float16)
    y=torch.empty(m,n,device='cuda',dtype=torch.float16)
    def old():
        return torch.nn.functional.linear(x,expand(w,s,x.dtype))
    old()
    print(json.dumps(dict(m=m,n=n,k=k,kind='expanded',ms=triton.testing.do_bench_cudagraph(old,rep=100))),flush=True)
    for bm,bn,bk,warps,stages in ((32,64,64,4,3),(32,64,128,4,3),(64,64,64,4,3),
                                (64,64,64,8,3),(128,64,64,8,2),(128,64,64,8,3)):
        def run():
            return _matmul[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,w,s,w,y,m,n,k,bm,bn,bk,False,num_warps=warps,num_stages=stages)
        try:
            run()
        except triton.runtime.errors.OutOfResources:
            continue
        print(json.dumps(dict(m=m,n=n,k=k,kind='tensorcore',bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,ms=triton.testing.do_bench_cudagraph(run,rep=100))),flush=True)
