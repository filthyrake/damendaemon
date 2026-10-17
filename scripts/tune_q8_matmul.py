"""Measure Q8 Tensor Core tiles against the established decode GEMV."""
import json
import torch
import triton
from damendaemon.quantized_kernels import _linear, _matmul

for m,n,k in ((4,28672,4096),(64,28672,4096),(4,4096,14336),(4,151936,896)):
    x=torch.randn(m,k,device='cuda',dtype=torch.float16)
    w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
    s=torch.full((n,k//32),.001,device='cuda',dtype=torch.float16)
    y=torch.empty(m,n,device='cuda',dtype=torch.float16)
    if m<=4:
        bn=2 if triton.next_power_of_2(k)>=4096 else 8
        def old():
            return _linear[(triton.cdiv(n,bn),)](x,w,s,w,y,k,n,triton.next_power_of_2(k),bn,m,False,num_warps=4,enable_fp_fusion=False)
        old()
        print(json.dumps(dict(m=m,n=n,k=k,kind='gemv',ms=triton.testing.do_bench_cudagraph(old,rep=100))),flush=True)
    for bm,bn,bk,warps in ((16,32,128,4),(16,64,128,4),(16,64,64,4),(16,128,64,4),(16,128,128,8)):
        def run():
            return _matmul[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,w,s,w,y,m,n,k,bm,bn,bk,False,num_warps=warps,num_stages=3)
        run()
        print(json.dumps(dict(m=m,n=n,k=k,kind='tensorcore',bm=bm,bn=bn,bk=bk,warps=warps,ms=triton.testing.do_bench_cudagraph(run,rep=100))),flush=True)
