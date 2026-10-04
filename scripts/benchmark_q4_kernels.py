"""Compare native packed Q4 kernels against a preserved pre-change module."""
import argparse
import importlib.util
import json
import sys
import torch
import triton as tr
import triton.testing
from dllm.runtime import require_gpu
from dllm.quantization import quantize_q4,dequantize_q4
from dllm import quantized_kernels_q4 as current
from dllm.quantized_kernels_q4 import linear


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    args=p.parse_args()
    require_gpu();torch.set_num_threads(4);torch.manual_seed(43)
    spec=importlib.util.spec_from_file_location('q4_baseline',args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    reports=[]
    for n,k in ((6144,4096),(28672,4096),(4096,14336),(32000,4096)):
        source=torch.randn(n,k,device='cuda',dtype=torch.float16)*.02
        weight,scales=quantize_q4(source)
        del source
        restored=dequantize_q4(weight,scales,torch.float16)
        for rows in (1,2,4,23):
            x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
            expected=torch.nn.functional.linear(x,restored)
            entry={'outputs':n,'width':k,'rows':rows}
            for name,fn in (('before',baseline.linear),('after',linear)):
                actual=fn(x,weight,scales)
                relative=((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                assert relative<.001,(entry,name,relative)
                entry[name+'_relative_rms']=relative
                entry[name+'_ms']=triton.testing.do_bench_cudagraph(lambda:fn(x,weight,scales),rep=100)
                module=baseline if name=='before' else current
                if rows==1 and name=='before':
                    bk=tr.next_power_of_2(k);bn=2 if bk>=4096 else 8
                    compiled=module._gemv[(tr.cdiv(n,bn),)](x,weight,scales,weight,actual,n,k,rows,bn,bk,False,
                        num_warps=4,enable_fp_fusion=False)
                else:
                    bm=16 if rows<=16 else tr.next_power_of_2(rows);bk=128 if rows<=16 or name=='after' else 64
                    compiled=module._gemm[(tr.cdiv(rows,bm),tr.cdiv(n,64))](x,weight,scales,weight,actual,
                        rows,n,k,bm,64,bk,False,num_warps=8 if bm==128 else 4,num_stages=3)
                entry[name+'_registers']=compiled.n_regs
                entry[name+'_spills']=compiled.n_spills
            print(json.dumps(entry),file=sys.stderr,flush=True)
            reports.append(entry)
        del restored,weight,scales
    print(json.dumps({'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'scope':'warm CUDA-graph matrix kernels; not whole-model serving','checks':reports},indent=2))


if __name__=='__main__':main()
