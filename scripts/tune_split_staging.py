"""Compare per-matrix activation splitting with repeated per-weight-tile splitting."""
import json
import torch
import triton as tr
from dllm.kernels import linear_fp32 as baseline


from dllm.precise_matmul import _split as split, _matmul as matmul


for m,n,k in ((2,6144,1024),(4,4096,1024),(4,6144,1024),(4,1024,3072),(4,151936,1024),(26,6144,1024),(104,6144,1024)):
    x=torch.randn(m,k,device='cuda')
    w=torch.randn(n,k,device='cuda',dtype=torch.float16)*.02
    hi,lo=torch.empty_like(x,dtype=torch.float16),torch.empty_like(x,dtype=torch.float16)
    scales=torch.empty(m,device='cuda')
    y=torch.empty(m,n,device='cuda')
    expected=torch.nn.functional.linear(x,w.float())
    print(json.dumps(dict(m=m,n=n,k=k,kind='baseline',ms=tr.testing.do_bench_cudagraph(lambda:baseline(x,w),rep=100))),flush=True)
    for bm,bn,bk,warps in ((16,64,64,4),(16,64,128,4),(16,128,64,4),(32,64,64,4),(64,64,64,4)):
        def run():
            split[(m,)](x,hi,lo,scales,k,tr.next_power_of_2(k))
            matmul[(tr.cdiv(m,bm),tr.cdiv(n,bn))](hi,lo,scales,w,y,m,n,k,bm,bn,bk,num_warps=warps,num_stages=3)
        run()
        relative=((y-expected).square().mean()/expected.square().mean()).sqrt().item()
        print(json.dumps(dict(m=m,n=n,k=k,kind='staged',bm=bm,bn=bn,bk=bk,warps=warps,relative_rms=relative,ms=tr.testing.do_bench_cudagraph(run,rep=100))),flush=True)
