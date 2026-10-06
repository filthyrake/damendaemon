"""Fixed #68 packed-compensated versus existing expanded-FP32 prefill control.

Synthetic packed matrices use pinned Qwen3.8-27B projection dimensions. Expansion
is inside timed calls; no resident dense cache. Selected columns use independent
CPU FP64 products from ordinary Torch Q4 coefficient decoding.
"""
import json
import os
import time
import torch
from dllm.compensated_q8 import linear_q4
from dllm.quantized_kernels_q4 import linear as expanded
from dllm.quantization import dequantize_q4
from dllm.rocm_policy import memory_limit
from dllm.runtime import require_gpu


@torch.inference_mode()
def main():
    if os.environ.get('DLLM_DEVICE')!='rocm' or memory_limit()!=2*1024**3:
        raise ValueError('requires ROCm and a 2GiB allocator cap')
    require_gpu();torch.set_num_threads(4);torch.manual_seed(68)
    for name,outputs,width in (('gate_up',34816,5120),('down',5120,17408),('qkv',14336,5120)):
        weight=torch.randint(0,256,(outputs,width//2),device='cuda',dtype=torch.uint8)
        scales=torch.full((outputs,width//32),.002,device='cuda',dtype=torch.float16)
        columns=torch.tensor([0,1,outputs//2,outputs-2,outputs-1],device='cuda')
        coefficients=dequantize_q4(weight[columns].cpu(),scales[columns].cpu()).double()
        for rows in (127,512):
            x=torch.randn(rows,width,device='cuda')
            expected=x.cpu().double() @ coefficients.T
            for route in ('compensated','expanded_fp32'):
                function=linear_q4 if route=='compensated' else expanded
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                value=function(x,weight,scales);torch.cuda.synchronize()
                delta=value[:,columns].cpu().double()-expected
                absolute=delta.abs().max().item()
                rms=(delta.square().mean()/expected.square().mean()).sqrt().item()
                print(json.dumps(dict(event='numerical',projection=name,rows=rows,route=route,
                    max_absolute=absolute,relative_rms=rms,passed=absolute<1e-4 and rms<1e-4,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())),flush=True)
                if absolute>=1e-4 or rms>=1e-4:raise AssertionError('numerical gate failed')
                del value,delta
            for round_index in range(2):
                for route in ('compensated','expanded_fp32','expanded_fp32','compensated'):
                    function=linear_q4 if route=='compensated' else expanded
                    torch.cuda.synchronize();start=time.perf_counter()
                    value=function(x,weight,scales);torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start;del value
                    print(json.dumps(dict(event='timing',projection=name,rows=rows,route=route,
                        round=round_index,seconds=elapsed)),flush=True)
            del x,expected
        del weight,scales,columns,coefficients
        torch.cuda.empty_cache()


if __name__=='__main__':main()
