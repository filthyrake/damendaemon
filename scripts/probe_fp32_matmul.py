"""Independent FP64 oracle and device timings for TF32x3 projections."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import statistics
import torch
from torch.nn import functional as F
from dllm.runtime import require_gpu
from fp32_matmul_experiment import linear


@torch.inference_mode()
def main():
    require_gpu();torch.manual_seed(431);torch.set_num_threads(4);checks=[]
    for m,n,k in ((1,7,32),(3,65,96),(17,129,257),(128,3336,768),(1536,3336,768),(1536,768,1536)):
        x=torch.randn(m,k);w=torch.randn(n,k)*.03;bias=torch.randn(n)*.01
        x[:,0]=1.e5
        expected=F.linear(x.double(),w.double(),bias.double())
        x,w,bias=x.cuda(),w.cuda(),bias.cuda()
        actual=linear(x,w,bias).cpu().double()
        relative=((actual-expected).square().mean()/expected.square().mean()).sqrt().item()
        assert relative<.00001,(m,n,k,relative)
        torch.testing.assert_close(actual,expected,atol=.01,rtol=.0001)
        timings={'cublas':[],'tf32x3':[]}
        for label,operation in [('cublas',F.linear),('tf32x3',linear),('tf32x3',linear),('cublas',F.linear)]*3:
            operation(x,w,bias);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):operation(x,w,bias)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):graph.replay()
            end.record();end.synchronize();timings[label].append(begin.elapsed_time(end)/10)
        checks.append({'rows':m,'outputs':n,'width':k,'relative_rms':relative,
            'max_absolute':(actual-expected).abs().max().item(),
            'median_ms':{k:statistics.median(v) for k,v in timings.items()},'trials_ms':timings})
    print(json.dumps({'passed':True,'checks':checks,'reference':'CPU FP64 on identical FP32 values, including inputs above FP16 range'},indent=2))


if __name__=='__main__':main()
