"""Independent matrix/graph gates for the isolated residual-int8 experiment."""
import json
import numpy as np
import torch
from experiments.kquant_residual import linear
from verify_kquant_kernels import fixture


def check(actual,expected,**case):
    error=actual.float()-expected.float()
    rms=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
    row=dict(**case,relative_rms=rms,max_absolute=error.abs().max().item())
    print(json.dumps(row),flush=True)
    assert torch.isfinite(actual).all() and rms<.001,row


def main():
    torch.set_num_threads(4);torch.manual_seed(513);rng=np.random.default_rng(513)
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():
        for width in (256,768,2048,8192):
            for kinds in ((12,),(13,),(14,),(14,12,13)):
                parts=[];values=[];spec=[];first=0;offset=0
                for i,kind in enumerate(kinds):
                    rows=17+i
                    data,expected=fixture(rng,kind,rows,width)
                    parts.append(data);values.append(expected)
                    spec.append((first,first+rows,kind,offset));first+=rows;offset+=data.nbytes
                weight=torch.from_numpy(np.concatenate(parts)).cuda()
                reference=torch.from_numpy(np.concatenate(values)).cuda().half()
                layout=(width,tuple(spec))
                for rows in (1,2,3,4):
                    x=torch.randn((rows,width),device='cuda',dtype=torch.float16)
                    bias=None if rows==1 else torch.randn(first,device='cuda',dtype=torch.float16)
                    check(linear(x,weight,layout,bias),torch.nn.functional.linear(x,reference,bias),
                          width=width,kinds=kinds,rows=rows,case='random')
                x=torch.randn((4,width),device='cuda',dtype=torch.float16)
                x.mul_(torch.logspace(-4,1,width,device='cuda',dtype=torch.float32).half())
                for _ in range(3):linear(x,weight,layout)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=linear(x,weight,layout)
                x.mul_(.3);graph.replay()
                check(actual,torch.nn.functional.linear(x,reference),width=width,kinds=kinds,rows=4,case='changed-input graph')
                x.zero_();graph.replay()
                assert torch.equal(actual,torch.zeros_like(actual))
                del graph,actual
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'scope':'experimental matrix and changed-input graph proof'}))


if __name__=='__main__':main()
