"""Real CUDA matrix gates across stage dispatch boundaries, without timing claims."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from dllm.q4_stage import StageQ4Linear
from dllm.quantization import quantize_q4,dequantize_q4


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    report=dict(passed=False,checks=[],scope='Synthetic real CUDA coefficient/layout and row-boundary correctness; not model performance',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('dllm/q4_stage.py','dllm/q4_stage_kernels.py')})
    torch.set_num_threads(4);torch.manual_seed(43)
    try:
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                for suffix,n,k in (('qkv.weight',1152,896),('out',896,896),('gate_up',9728,896),('down',896,4864)):
                    packed,scales=quantize_q4(torch.randn(n,k,device='cuda',dtype=torch.float16)*.02)
                    restored=dequantize_q4(packed,scales,torch.float16)
                    base=SimpleNamespace(scales={id(packed):scales})
                    operation=StageQ4Linear(base,{'0.'+suffix:packed})
                    bias=torch.randn(n,device='cuda',dtype=torch.float16)*.01 if suffix=='qkv.weight' else None
                    for rows in (1,2,4,16,17,32,64,128,129,256,512):
                        x=torch.randn(rows,k,device='cuda',dtype=torch.float16)
                        expected=torch.nn.functional.linear(x,restored,bias)
                        actual=operation(x,packed,bias)
                        error=((actual.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                        exact=bool(torch.equal(actual,expected))
                        item=dict(device=torch.cuda.get_device_name(),projection=suffix,rows=rows,relative_rms=error,
                                  exact_dense=exact,passed=error<.001 and (rows<=128 or exact))
                        report['checks'].append(item)
                        assert item['passed'],item
                    del operation,base,packed,scales,restored,x,actual,expected,bias
                    torch.cuda.synchronize()
                print(torch.cuda.get_device_name(),'matrix gates passed',flush=True)
        assert report['checks'],'no CUDA devices tested'
        report['passed']=True
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
