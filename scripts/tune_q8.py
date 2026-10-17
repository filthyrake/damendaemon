"""Developer-only A2 Q8 layout experiment; measures kernels, not LLM throughput."""
import json
import torch
import triton
from damendaemon.quantized_kernels import _linear

torch.manual_seed(11)
reports=[]
for k,n in ((896,151936),(896,4864),(4864,896)):
    w=torch.randint(-127,128,(n,k),device='cuda',dtype=torch.int8)
    s=torch.full((n,k//32),.001,device='cuda',dtype=torch.float16)
    for rows in (1,4):
        x=torch.randn((rows,k),device='cuda',dtype=torch.float16)
        y=torch.empty((rows,n),device='cuda',dtype=torch.float16)
        for bn in (1,2,4,8,16,32):
            for warps in (4,8,16):
                def run():
                    return _linear[(triton.cdiv(n,bn),)](x,w,s,w,y,k,n,triton.next_power_of_2(k),bn,rows,False,num_warps=warps,enable_fp_fusion=False)
                try:
                    kernel=run()
                except triton.runtime.errors.OutOfResources:
                    continue
                ms=triton.testing.do_bench_cudagraph(run,rep=100)
                report=dict(k=k,n=n,rows=rows,bn=bn,warps=warps,ms=ms,registers=kernel.n_regs,spills=kernel.n_spills)
                reports.append(report)
                print(json.dumps(report),flush=True)
