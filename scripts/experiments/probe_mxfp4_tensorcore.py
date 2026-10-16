if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import os
from pathlib import Path
import statistics
import sys
import torch
from safetensors import safe_open
from dllm import mxfp4_kernels as base
import mxfp4_tensorcore as candidate

if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')

torch.set_num_threads(4);torch.manual_seed(17)
root=Path(sys.argv[1]);catalog=json.loads((root/'dllm-source.json').read_text())['tensors']
def get(key):
    with safe_open(str(root/catalog[key]['file']),framework='pt',device='cpu') as file:return file.get_tensor(key).cuda()
rows=[]
with torch.inference_mode():
    for layer in (0,23):
        for projection in ('gate_up_proj','down_proj'):
            name=f'model.layers.{layer}.mlp.experts.'+projection
            w=get(name+'_blocks');s=get(name+'_scales');bias=get(name+'_bias').float()
            assert bool(((s>=104)&(s<=140)).all());e,n,g,_=w.shape;k=g*32
            for tokens in (1,3):
                shared=projection=='gate_up_proj';selected=torch.randint(e,(tokens,4),device='cuda')
                x=torch.randn(tokens if shared else tokens*4,k,device='cuda')
                functions=[lambda:base.routed_linear(x,w,s,selected,bias,shared),lambda:candidate.routed_linear(x,w,s,selected,bias,shared)]
                outputs=[f() for f in functions]
                error=outputs[1]-outputs[0];rms=float(error.square().mean().sqrt()/outputs[0].square().mean().sqrt())
                assert rms<1e-5 and float(error.abs().max())<2e-3
                graphs=[]
                for f in functions:
                    for _ in range(3):f()
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=f()
                    graphs.append((graph,output))
                times=[[],[]]
                for trial in range(12):
                    for which in ((0,1) if trial%2==0 else (1,0)):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(16):graphs[which][0].replay()
                        end.record();end.synchronize();times[which].append(start.elapsed_time(end)/16)
                medians=[statistics.median(t) for t in times]
                row={'layer':layer,'projection':projection,'tokens':tokens,'relative_rms':rms,'max_absolute_error':float(error.abs().max()),'median_ms':medians,'speedup':medians[0]/medians[1],'milliseconds':times}
                rows.append(row);print(json.dumps(row),flush=True)
result={'passed':True,'gpu':torch.cuda.get_device_name(),'rows':rows}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
