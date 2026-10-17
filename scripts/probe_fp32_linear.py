"""Measure FP32 projection launch geometry without changing runtime dispatch."""
import json
import torch
import triton
from damendaemon.kernels import _linear_fp32
from damendaemon.runtime import require_gpu


@torch.inference_mode()
def main():
    require_gpu();torch.manual_seed(12);results=[]
    for rows in (1,4):
        for n,k in ((1152,512),(512,384),(2048,512),(512,1024),(32128,512)):
            x=torch.randn(rows,k,device='cuda');w=torch.randn(n,k,device='cuda')
            expected=torch.nn.functional.linear(x,w)
            for bn in (4,8,16,32):
                for warps in (4,8):
                    out=torch.empty_like(expected)
                    def operation():
                        _linear_fp32[(triton.cdiv(n,bn),)](x,w,out,k,n,triton.next_power_of_2(k),bn,rows,num_warps=warps,enable_fp_fusion=False)
                    operation();torch.testing.assert_close(out,expected,atol=.0001,rtol=.0001)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(20):operation()
                    for _ in range(3):graph.replay()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):graph.replay()
                    end.record();end.synchronize()
                    results.append({'rows':rows,'n':n,'k':k,'outputs_per_block':bn,'warps':warps,'microseconds':start.elapsed_time(end)*1000/200})
    print(json.dumps({'passed':True,'results':results},indent=2))


if __name__=='__main__':main()
