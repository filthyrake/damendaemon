"""Compare fused Q4 prefill tiles with production expansion plus dense GEMM."""
import argparse
import json
from pathlib import Path

import torch
import triton as tr
import triton.testing
from dllm.runtime import require_gpu
from dllm.quantization import quantize_q4, dequantize_q4
from dllm.quantized_kernels_q4 import linear, _gemm


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--single-dot',action='store_true');args=p.parse_args()
    kernel=_gemm
    if args.single_dot:
        from experiments.q4_prefill import single_dot
        kernel=single_dot
    require_gpu();torch.set_num_threads(4);torch.manual_seed(417)
    torch.backends.cuda.matmul.allow_tf32=False
    report=dict(kernel="single-dot" if args.single_dot else "paired-dot", device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],
                scope='synthetic Q4 prefill; existing FP16 coefficient policy; no production dispatch changes')
    tiles=((32,64,128,4),(64,64,128,4),(128,64,128,8),(64,128,128,8),(64,64,256,4))
    try:
        for n,k in ((6144,4096),(4096,4096),(28672,4096),(4096,14336),(113,4096)):
            weights,scales=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
            decoded=dequantize_q4(weights,scales,torch.float16)
            bias=torch.randn(n,device='cuda',dtype=torch.float16)
            for rows in (129,258,400):
                x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
                expected=torch.nn.functional.linear(x,decoded,bias);out=torch.empty_like(expected)
                case=dict(outputs=n,width=k,rows=rows,candidates=[],
                          production_ms=triton.testing.do_bench_cudagraph(lambda:linear(x,weights,scales,bias),rep=100))
                report['cases'].append(case)
                for bm,bn,bk,warps in tiles:
                    item=dict(bm=bm,bn=bn,bk=bk,warps=warps);case['candidates'].append(item)
                    try:
                        def launch():
                            return kernel[(tr.cdiv(rows,bm),tr.cdiv(n,bn))](x,weights,scales,bias,out,
                                rows,n,k,bm,bn,bk,True,num_warps=warps,num_stages=3)
                        compiled=launch();error=out.float()-expected.float()
                        rms=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        item.update(relative_rms=rms,max_absolute=error.abs().max().item(),
                                    registers=compiled.n_regs,spills=compiled.n_spills,
                                    passed=bool(torch.isfinite(out).all()) and rms<.001)
                        if item['passed']:item['ms']=triton.testing.do_bench_cudagraph(launch,rep=100)
                    except Exception as error:item.update(passed=False,error=repr(error))
                good=[c for c in case['candidates'] if c.get('passed')]
                print(json.dumps(dict(outputs=n,width=k,rows=rows,production_ms=case['production_ms'],
                    best=min(good,key=lambda c:c['ms']) if good else None)),flush=True)
            del weights,scales,decoded
        report['passed']=all(c.get('passed') for case in report['cases'] for c in case['candidates'])
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
