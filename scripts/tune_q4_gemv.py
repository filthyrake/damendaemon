"""A2-only tile sweep; every candidate must meet the independent matrix gate."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import torch
import triton as tr
import triton.language as tl
import triton.testing
from dllm.runtime import require_gpu
from dllm.quantization import quantize_q4,dequantize_q4
from dllm.quantized_kernels_q4 import _gemv,linear

@tr.jit
def _reduce_parts(T,BIAS,Y,N:tl.constexpr,ROWS:tl.constexpr,PARTS:tl.constexpr,
                  BP:tl.constexpr,BIASED:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    parts=tl.arange(0,BP)
    values=tl.load(T+offsets[:,None]*PARTS+parts[None,:],
                   (offsets[:,None]<N*ROWS)&(parts[None,:]<PARTS),other=0)
    result=tl.sum(values,1)
    if BIASED:result+=tl.load(BIAS+offsets%N,offsets<N*ROWS,other=0).to(tl.float32)
    tl.store(Y+offsets,result,offsets<N*ROWS)


def split_gemv(x,weight,scales,bias=None,bk=1024,bn=4,warps=4):
    rows,k=x.shape;n=weight.shape[0];parts=tr.cdiv(k,bk)
    partial=torch.empty((rows,n,parts),device=x.device,dtype=torch.float32)
    result=torch.empty((rows,n),device=x.device,dtype=x.dtype)
    _gemv[(tr.cdiv(n,bn),parts)](x,weight,scales,weight,partial,n,k,rows,bn,bk,False,parts,
                               num_warps=warps,enable_fp_fusion=False)
    _reduce_parts[(tr.cdiv(rows*n,128),)](partial,weight if bias is None else bias,result,n,rows,parts,
        tr.next_power_of_2(parts),bias is not None,128,num_warps=4)
    return result



require_gpu();torch.set_num_threads(4);torch.manual_seed(71)
reports=[]
for n,k in ((6144,4096),(4096,14336)):
    q,s=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
    restored=dequantize_q4(q,s,torch.float16)
    for rows in (1,2,4):
        x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
        expected=torch.nn.functional.linear(x,restored)
        candidates=[]
        baseline=triton.testing.do_bench_cudagraph(lambda:linear(x,q,s),rep=50)
        for bk in (512,1024,2048):
            for bn,warps in ((1,4),(2,4),(4,4),(8,4),(4,8)):
                value=split_gemv(x,q,s,bk=bk,bn=bn,warps=warps)
                relative=((value.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                assert relative<.001,(n,k,rows,bk,bn,warps,relative)
                ms=triton.testing.do_bench_cudagraph(lambda:split_gemv(x,q,s,bk=bk,bn=bn,warps=warps),rep=50)
                candidates.append({'bk':bk,'bn':bn,'warps':warps,'ms':ms,'relative_rms':relative})
        result={'outputs':n,'width':k,'rows':rows,'unsplit_ms':baseline,'candidates':candidates}
        reports.append(result)
        print(json.dumps({'outputs':n,'width':k,'rows':rows,'unsplit_ms':baseline,'best':min(candidates,key=lambda c:c['ms'])}),flush=True)
open('q4-split-tuning.json','w').write(json.dumps({'device':torch.cuda.get_device_name(),'results':reports},indent=2))
