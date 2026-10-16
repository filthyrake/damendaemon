"""Packed projection versus temporary expansion for short Metal prompt matrices."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import statistics
import time
import torch
from unittest.mock import patch
from dllm import quantized_metal,quantized_metal_q4


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quantization',choices=('q8_0','q4_0'),default='q8_0')
    args=parser.parse_args();torch.manual_seed(419);torch.set_num_threads(4)
    module=quantized_metal if args.quantization=='q8_0' else quantized_metal_q4
    shader=module.library();results=[]
    with patch.object(module,'use_packed_projection',lambda rows,outputs,width,fp32:rows<=4):
        for outputs,width in ((1152,896),(896,896),(9728,896),(896,4864)):
            w=(torch.randint(-127,128,(outputs,width),dtype=torch.int8,device='mps') if args.quantization=='q8_0' else
               torch.randint(0,256,(outputs,width//2),dtype=torch.uint8,device='mps'))
            scales=torch.full((outputs,width//32),.01,dtype=torch.float16,device='mps')
            for rows in (8,16,32,64):
                x=torch.randn(rows,width,device='mps')
                def run(mode):
                    if mode=='expanded':return module.linear(x,w,scales)
                    out=torch.empty((rows,outputs),device='mps')
                    kernel=shader.linear_index32 if args.quantization=='q8_0' else shader.linear_float_index32
                    kernel(x,w,scales,x,out,rows,outputs,width,0,
                        threads=(((outputs+7)//8)*256,(rows+3)//4),group_size=(256,1))
                    return out
                expected=run('expanded');actual=run('packed')
                relative=((actual-expected).square().mean()/expected.square().mean()).sqrt().item()
                assert relative<1e-5 and torch.isfinite(actual).all(),relative
                timings={label:[] for label in ('expanded','packed')}
                for label in ('expanded','packed','packed','expanded')*3:
                    for _ in range(3):run(label)
                    torch.mps.synchronize();start=time.perf_counter()
                    for _ in range(20):run(label)
                    torch.mps.synchronize();timings[label].append((time.perf_counter()-start)*1000/20)
                results.append(dict(rows=rows,outputs=outputs,width=width,relative_rms=relative,
                    median_ms={k:statistics.median(v) for k,v in timings.items()},trials_ms=timings))
    print(json.dumps(dict(passed=True,device='Apple M4',quantization=args.quantization,
        scope='Identical quantized coefficients, FP32 activations; packed versus MPS matmul accumulation differs; component experiment only',results=results),indent=2))


if __name__=='__main__':main()
