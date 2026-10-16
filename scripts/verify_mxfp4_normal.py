"""Independent exact value-bit and routed-graph proof for validated normal scales."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import os
import torch
import triton as tr
import triton.language as tl
from dllm.mxfp4 import routed_linear
from dllm.mxfp4_kernels import _value_normal


@tr.jit
def values(Y,N:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    code=index%16;scale=index//16+2
    tl.store(Y+index,_value_normal(code,scale),index<N)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4);torch.manual_seed(67)
    table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
    expected=torch.ldexp(table[None,:],torch.arange(2,253)[:,None]-127).float().flatten()
    actual=torch.empty_like(expected,device='cuda');values[(tr.cdiv(actual.numel(),256),)](actual,actual.numel(),256)
    assert torch.equal(actual.cpu().view(torch.int32),expected.view(torch.int32))
    rows=[]
    with torch.inference_mode():
        for count,width,outputs in ((1,96,37),(3,2880,257)):
            blocks=torch.randint(0,256,(5,outputs,width//32,16),device='cuda',dtype=torch.uint8)
            scales=torch.randint(104,141,blocks.shape[:-1],device='cuda',dtype=torch.uint8)
            selected=torch.randint(5,(count,3),device='cuda');bias=torch.randn(5,outputs,device='cuda')
            for shared in (True,False):
                selected=torch.randint(5,(count,3),device='cuda')
                x=torch.randn(count if shared else count*3,width,device='cuda')
                def project(normal):return routed_linear(x,blocks,scales,selected,bias,shared,normal_scales=normal)
                expected=project(False);actual=project(True);assert torch.equal(actual,expected)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):project(True)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=project(True)
                for offset in (1,2):
                    selected.add_(offset).remainder_(5);graph.replay();assert torch.equal(captured,project(False))
                for first,second in ((-1,5),(1<<32,-(1<<32))):
                    selected[0,0]=first;selected[0,1]=second;graph.replay();assert torch.isnan(captured[:2]).all()
                rows.append({'tokens':count,'width':width,'outputs':outputs,'shared':shared,'exact':True,'live_graph_and_invalid_bounds':True})
    print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'all_4016_normal_value_bits_exact':True,
        'independent_reference':'CPU float64 E2M1 table and ldexp, including signed zero and normal range boundaries','rows':rows},indent=2))


if __name__=='__main__':main()
