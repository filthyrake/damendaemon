"""FP32-storage tensor-core projection experiment; never runtime dispatch."""
import json
import torch
import triton as tr
import triton.language as tl
from damendaemon.kernels import linear_fp32
from damendaemon.runtime import require_gpu


@tr.jit
def projection(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.arange(0,16)
    outputs=tl.program_id(0)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    acc=tl.full((16,BN),0,tl.float32)
    for block in range(tr.cdiv(K,BK)):
        kk=block*BK+columns
        x=tl.load(X+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),other=0)
        w=tl.load(W+outputs[None,:]*K+kk[:,None],(outputs[None,:]<N)&(kk[:,None]<K),other=0)
        acc+=tl.dot(x,w,input_precision='tf32x3')
    tl.store(Y+rows[:,None]*N+outputs[None,:],acc,(rows[:,None]<M)&(outputs[None,:]<N))


@torch.inference_mode()
def main():
    require_gpu();torch.manual_seed(31);results=[]
    for rows in (1,4):
        for n,k in ((1152,512),(512,384),(2048,512),(512,1024),(32128,512)):
            x=torch.randn(rows,k,device='cuda');w=torch.randn(n,k,device='cuda');out=torch.empty(rows,n,device='cuda')
            expected=torch.nn.functional.linear(x.cpu().double(),w.cpu().double()).float().cuda()
            candidates=[('current',lambda:linear_fp32(x,w)),('torch',lambda:torch.nn.functional.linear(x,w))]
            for bn in (16,32,64):
                for bk in (32,64):
                    def operation(bn=bn,bk=bk):
                        projection[(tr.cdiv(n,bn),)](x,w,out,rows,n,k,bn,bk,num_warps=4)
                        return out
                    candidates.append((f'tf32x3-{bn}-{bk}',operation))
            for name,operation in candidates:
                actual=operation();torch.testing.assert_close(actual,expected,atol=.0001,rtol=.0001)
                error=(actual-expected).abs().max().item()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):operation()
                for _ in range(3):graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize()
                results.append({'rows':rows,'n':n,'k':k,'kernel':name,'max_absolute':error,'microseconds':start.elapsed_time(end)*1000/200})
    print(json.dumps({'passed':True,'results':results},indent=2))


if __name__=='__main__':main()
