"""Measure contiguous packed loads with paired low/high Tensor Core products for Q4 matrix products."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import torch
import triton as tr
import triton.language as tl
import triton.testing
from dllm.runtime import require_gpu
from dllm.quantization import quantize_q4,dequantize_q4
from dllm.quantized_kernels_q4 import linear


@tr.jit
def candidate(X,W,S,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    outputs=tl.program_id(1)*BN+tl.arange(0,BN)
    columns=tl.arange(0,BK)
    packed_columns=tl.arange(0,BK//2)
    groups=tl.arange(0,BK//32)
    result=tl.full((BM,BN),0,tl.float32)
    for start in range(tr.cdiv(K,BK)):
        k=start*BK+(packed_columns//16)*32+packed_columns%16
        xlow=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        xhigh=tl.load(X+rows[:,None]*K+k[None,:]+16,(rows[:,None]<M)&(k[None,:]+16<K),other=0)
        p=start*(BK//2)+packed_columns
        packed=tl.load(W+outputs[:,None]*(K//2)+p[None,:],(outputs[:,None]<N)&(p[None,:]<K//2),other=0).to(tl.int32)
        g=start*(BK//32)+groups
        scales=tl.load(S+outputs[:,None]*(K//32)+g[None,:],(outputs[:,None]<N)&(g[None,:]<K//32),other=0).to(tl.float32)
        low=(((packed&15)-8).to(tl.float32).reshape(BN,BK//32,16)*scales[:,:,None]).reshape(BN,BK//2).to(tl.float16)
        high=(((packed>>4)-8).to(tl.float32).reshape(BN,BK//32,16)*scales[:,:,None]).reshape(BN,BK//2).to(tl.float16)
        result+=tl.dot(xlow,tl.trans(low))
        result+=tl.dot(xhigh,tl.trans(high))
    tl.store(Y+rows[:,None]*N+outputs[None,:],result,(rows[:,None]<M)&(outputs[None,:]<N))


def main():
    require_gpu();torch.set_num_threads(4);torch.manual_seed(91)
    reports=[]
    for n,k in ((6144,4096),(4096,14336)):
        q,s=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
        restored=dequantize_q4(q,s,torch.float16)
        for rows in (1,4,23):
            x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
            expected=torch.nn.functional.linear(x,restored)
            y=torch.empty_like(expected)
            baseline=triton.testing.do_bench_cudagraph(lambda:linear(x,q,s),rep=50)
            candidates=[]
            bm=16 if rows<=16 else 32
            for bn,bk,warps in ((32,128,4),(64,128,4),(64,256,4),(128,128,4),(64,128,8)):
                def launch():
                    return candidate[(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,q,s,y,rows,n,k,bm,bn,bk,num_warps=warps,num_stages=3)
                compiled=launch()
                relative=((y.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                assert relative<.001,(n,k,rows,bn,bk,relative)
                ms=triton.testing.do_bench_cudagraph(launch,rep=50)
                candidates.append({'bn':bn,'bk':bk,'warps':warps,'ms':ms,'relative_rms':relative,'registers':compiled.n_regs,'spills':compiled.n_spills})
            record={'outputs':n,'width':k,'rows':rows,'previous_ms':baseline,'candidates':candidates}
            reports.append(record)
            print(json.dumps({'outputs':n,'width':k,'rows':rows,'previous_ms':baseline,'best':min(candidates,key=lambda c:c['ms'])}),flush=True)
    open('q4-contiguous-tuning.json','w').write(json.dumps({'device':torch.cuda.get_device_name(),'results':reports},indent=2))


if __name__=='__main__':main()
