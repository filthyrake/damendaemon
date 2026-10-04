"""Bounded native projection tile probe on the authorized A2/A30."""
import argparse
import json
import os
import statistics
import time

import torch
import triton as tr
from dllm.precise_matmul import _split,_matmul,linear


CONFIGS={
    'baseline':(64,64,64,4,3),
    'stages2':(64,64,64,4,2),
    'm32':(32,64,64,4,3),
    'm32n128':(32,128,64,4,2),
    'n128k32':(64,128,32,4,2),
    'n128':(64,128,64,4,2),
    'n128w8':(64,128,64,8,2),
    'n256':(64,256,32,8,2),
}
AUTHORIZED={'GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283','GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configs',nargs='+',choices=CONFIGS,default=list(CONFIGS))
    parser.add_argument('--rows',nargs='+',type=int,default=[129,512,2048])
    parser.add_argument('--held-out',action='store_true');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in AUTHORIZED:raise RuntimeError('restricted to authorized GPUs')
    torch.set_num_threads(4);torch.manual_seed(89);reports=[]
    def candidate(x,w,config):
        bm,bn,bk,warps,stages=config;m,k=x.shape;n=w.shape[0]
        hi=torch.empty_like(x,dtype=torch.float16);lo=torch.empty_like(hi)
        scales=torch.empty(m,device='cuda',dtype=torch.float32)
        out=torch.empty((m,n),device='cuda',dtype=torch.float32)
        _split[(m,)](x,hi,lo,scales,k,tr.next_power_of_2(k))
        _matmul[(tr.cdiv(m,bm),tr.cdiv(n,bn))](hi,lo,scales,w,out,m,n,k,bm,bn,bk,
            num_warps=warps,num_stages=stages)
        return out
    with torch.inference_mode():
        shapes=((1024,4096),(3072,1024),(2048,12288),(6144,2048)) if args.held_out else ((2048,3072),(2048,16384),(8192,2048),(1024,6144))
        for count in args.rows:
            for inputs,outputs in shapes:
                x=torch.randn(count,inputs,device='cuda');w=(torch.randn(outputs,inputs,device='cuda')*.02).half()
                reference=linear(x,w);rows=[]
                # Rotate order across shapes to reduce fixed-order bias.
                names=list(args.configs);shift=len(reports)%len(names);names=names[shift:]+names[:shift]
                for name in names:
                    config=CONFIGS[name]
                    try:
                        actual=candidate(x,w,config)
                        maximum=(actual-reference).abs().max().item()
                        torch.testing.assert_close(actual,reference,rtol=1e-4,atol=3e-5)
                        candidate(x,w,config);torch.cuda.synchronize();times=[]
                        for _ in range(7):
                            start=time.perf_counter();candidate(x,w,config);torch.cuda.synchronize()
                            times.append(1000*(time.perf_counter()-start))
                        rows.append({'config':name,'tile':config,'median_ms':statistics.median(times),
                            'milliseconds':times,'max_error_vs_existing':maximum})
                    except tr.OutOfResources as error:
                        rows.append({'config':name,'error':str(error)})
                reports.append({'rows':count,'inputs':inputs,'outputs':outputs,'trials':rows})
    print(json.dumps({'gpu':torch.cuda.get_device_name(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
        'torch':str(torch.__version__),'held_out':args.held_out,'reports':reports,
        'method':'Same compensated arithmetic and stored weights; seven synchronized wrapper timings after warmup. Tile variants are experimental, not runtime defaults.'},indent=2))


if __name__=='__main__':main()
