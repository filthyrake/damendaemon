"""Independent matrix/graph gates for the isolated residual-int8 experiment."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--float32-experimental',action='store_true');args=parser.parse_args()
    kernel=linear
    dtype=torch.float16
    if args.float32_experimental:
        from experiments.kquant_residual_f32 import linear as kernel
        dtype=torch.float32
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
                reference=torch.from_numpy(np.concatenate(values)).cuda().to(dtype)
                layout=(width,tuple(spec))
                for rows in (1,2,3,4):
                    x=torch.randn((rows,width),device='cuda',dtype=dtype)
                    bias=None if rows==1 else torch.randn(first,device='cuda',dtype=dtype)
                    check(kernel(x,weight,layout,bias),torch.nn.functional.linear(x,reference,bias),
                          width=width,kinds=kinds,rows=rows,case='random')
                x=torch.randn((4,width),device='cuda',dtype=dtype)
                x.mul_(torch.logspace(-4,1,width,device='cuda',dtype=torch.float32).to(dtype))
                for _ in range(3):kernel(x,weight,layout)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=kernel(x,weight,layout)
                x.mul_(.3);graph.replay()
                check(actual,torch.nn.functional.linear(x,reference),width=width,kinds=kinds,rows=4,case='changed-input graph')
                x.zero_();graph.replay()
                assert torch.equal(actual,torch.zeros_like(actual))
                del graph,actual
    if args.float32_experimental:
        from experiments.kquant_residual_f32 import library
        _,quantize,_=library(os.environ['DLLM_EXPERIMENTAL_RESIDUAL_F32_LIBRARY'])
        exponents=(-149,-140,-126,-120,-100,-50,0,50,100)
        cpu=torch.stack([torch.linspace(-.99,.99,32,dtype=torch.float64)*2.**e for e in exponents]).float()
        x=cpu.cuda()
        # Pad to the matrix ABI's 256-element row alignment.
        padded=torch.zeros((1,512),device='cuda',dtype=torch.float32)
        padded[0,:x.numel()]=x.flatten()
        first=torch.empty_like(padded,dtype=torch.int8);second=torch.empty_like(first)
        scales=torch.empty((16,4),device='cuda',dtype=torch.float32)
        error=quantize(padded.data_ptr(),first.data_ptr(),second.data_ptr(),scales.data_ptr(),
                       1,512,torch.cuda.current_stream().cuda_stream)
        assert error==0,error
        q1=first.cpu().double().reshape(16,32);q2=second.cpu().double().reshape(16,32)
        ds=scales.cpu().double()
        restored=q1*ds[:,0:1]+q2*ds[:,1:2]
        expected=padded.cpu().double().reshape(16,32)
        difference=(restored-expected).abs().amax(-1)
        bound=expected.abs().amax(-1)*.0001+2.**-149
        assert torch.isfinite(ds).all() and torch.all(difference<=bound),(difference,bound)
        print(json.dumps(dict(case='FP32 activation reconstruction',exponents=exponents,
                              max_absolute_errors=difference[:len(exponents)].tolist(),passed=True)))
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'scope':'experimental matrix and changed-input graph proof','dtype':str(dtype)}))


if __name__=='__main__':main()
