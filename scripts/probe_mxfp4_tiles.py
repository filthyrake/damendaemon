"""Compare routed MXFP4 output tiles on real checkpoint tensors."""
import argparse
import json
import os
import statistics
import time

import torch
import triton as tr
from safetensors import safe_open
from damendaemon.mxfp4_kernels import _routed
from damendaemon.mxfp4 import routed_linear


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    from pathlib import Path
    root=Path(args.model);catalog=json.loads((root/'damendaemon-source.json').read_text())['tensors']
    def get(key):
        with safe_open(str(root/catalog[key]['file']),framework='pt',device='cpu') as reader:
            return reader.get_tensor(key).cuda()
    rows=[];torch.manual_seed(104)
    with torch.inference_mode():
        for projection in ('gate_up_proj','down_proj'):
            name='model.layers.0.mlp.experts.'+projection
            w=get(name+'_blocks');s=get(name+'_scales');b=get(name+'_bias').float()
            e,n,g,_=w.shape;k=g*32
            for tokens in (1,3):
                shared=projection=='gate_up_proj';top=4
                selected=torch.randint(e,(tokens,top),device='cuda')
                x=torch.randn(tokens if shared else tokens*top,k,device='cuda')
                y=torch.empty((tokens*top,n),device='cuda');target=routed_linear(x,w,s,selected,b,shared)
                for bn in (1,2,4,8):
                    for warps in (4,8):
                        def run():
                            _routed[(tr.cdiv(n,bn),tokens*top)](x,w,s,selected,b,y,e,k,n,top,
                                shared,True,tr.next_power_of_2(k),bn,num_warps=warps,enable_fp_fusion=False)
                        for _ in range(3):run()
                        torch.testing.assert_close(y,target,rtol=2e-5,atol=2e-4)
                        torch.cuda.synchronize();times=[]
                        for _ in range(9):
                            start=time.perf_counter();run();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                        rows.append({'projection':projection,'tokens':tokens,'bn':bn,'warps':warps,
                            'median_ms':statistics.median(times),'milliseconds':times})
    print(json.dumps({'gpu':torch.cuda.get_device_name(),'rows':rows},indent=2))


if __name__=='__main__':main()
