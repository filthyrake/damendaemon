"""Qualify native packed blocks against upstream gguf-py before model dispatch."""
import argparse
import json
import importlib.metadata
import triton

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize
from dllm.runtime import require_gpu
from dllm.device import device_name
from dllm.kquant_kernels import expand, embedding, linear


def fixture(rng,kind,rows,width):
    block,size=GGML_QUANT_SIZES[GGMLQuantizationType(kind)]
    data=rng.integers(0,256,(rows*width//block,size),dtype=np.uint8)
    if kind==1:
        data=rng.normal(0,.1,(rows,width)).astype(np.float16).view(np.uint8)
    else:
        offsets=(0,2) if kind in (12,13) else (208,) if kind==14 else (0,)
        for offset in offsets:
            scale=rng.uniform(-.005,.005,len(data)).astype(np.float16)
            scale[0]=0
            data[:,offset:offset+2]=scale.view(np.uint8).reshape(-1,2)
    expected=dequantize(data,GGMLQuantizationType(kind)).reshape(rows,width)
    return data.reshape(-1),expected


def main():
    p=argparse.ArgumentParser();p.parse_args()
    require_gpu();torch.set_num_threads(4);torch.manual_seed(319)
    torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(319); reports=[]
    with torch.inference_mode():
        for width in (256,768,2048):
            for kinds in ((1,),(2,),(8,),(12,),(13,),(14,),(12,14,12),(1,8,13)):
                parts=[];values=[];spec=[];first=0;offset=0
                for index,kind in enumerate(kinds):
                    rows=35 if len(kinds)==1 else 17+index
                    data,expected=fixture(rng,kind,rows,width)
                    parts.append(data);values.append(expected)
                    spec.append((first,first+rows,kind,offset));first+=rows;offset+=data.nbytes
                weight=torch.from_numpy(np.concatenate(parts)).cuda()
                reference=torch.from_numpy(np.concatenate(values)).cuda()
                layout=(width,tuple(spec));record={'width':width,'kinds':kinds,'rows':first,'matmul':[]}
                for dtype in (torch.float16,torch.float32):
                    actual=expand(weight,layout,dtype)
                    assert torch.equal(actual,reference.to(dtype)),(width,kinds,'expand',dtype,float((actual-reference.to(dtype)).abs().max()))
                    ids=torch.tensor([first-1,0,first//2,1],device='cuda')
                    selected=embedding(ids,weight,layout,dtype)
                    assert torch.equal(selected,reference[ids].half().to(dtype)),(width,kinds,'embedding',dtype)
                    for _ in range(3):embedding(ids,weight,layout,dtype)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=embedding(ids,weight,layout,dtype)
                    ids.copy_(torch.tensor([0,first-1,1,first//2],device='cuda'));graph.replay()
                    assert torch.equal(output,reference[ids].half().to(dtype))
                    for batch in (1,4,17,129):
                        x=torch.randn(batch,width,device='cuda',dtype=dtype)
                        bias=None if batch in (1,17) else torch.randn(first,device='cuda',dtype=dtype)
                        expected=torch.nn.functional.linear(x,reference.to(dtype),bias)
                        actual=linear(x,weight,layout,bias)
                        error=actual.float()-expected.float()
                        absolute=error.abs().max().item()
                        relative=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
                        assert torch.isfinite(actual).all() and relative<.001,(width,kinds,batch,dtype,relative,absolute)
                        record['matmul'].append({'dtype':str(dtype),'batch':batch,'bias':bias is not None,'relative_rms':relative,'max_absolute':absolute})
                    del graph,output
                    staging=torch.randn(4,width,device='cuda',dtype=dtype)
                    for _ in range(3):linear(staging,weight,layout)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=linear(staging,weight,layout)
                    staging.mul_(.3);graph.replay()
                    expected=torch.nn.functional.linear(staging,reference.to(dtype))
                    relative=((output.float()-expected.float()).square().mean()/expected.float().square().mean()).sqrt().item()
                    assert relative<.001,(width,kinds,'linear graph',dtype,relative)
                    del graph,output
                record['bitwise_dequantization']=True;record['graph_lookup_changed_ids']=True;reports.append(record)
    print(json.dumps({'passed':True,'device':device_name(),'torch':torch.__version__,'triton':triton.__version__,'gguf':importlib.metadata.version('gguf'),'cases':reports},indent=2))


if __name__=='__main__':main()
