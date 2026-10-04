"""Correctness and counterbalanced device timing for block-SSD prefill."""
import json
import math
import statistics
import torch
from dllm.runtime import require_gpu
from dllm import mamba2_kernels,mamba2_ssd


@torch.inference_mode()
def main():
    require_gpu();torch.manual_seed(401);reports=[];known=[]
    for length in (128,129):
        x=torch.ones(1,length,1792,device='cuda');gate=torch.ones(1,length,1536,device='cuda')
        dt=torch.zeros(1,length,24,device='cuda');zero=torch.zeros(24,device='cuda')
        state=torch.zeros(1,1536,128,device='cuda');slots=torch.tensor([0],device='cuda')
        actual=mamba2_ssd.scan(x,gate,dt,zero,zero-1,zero,state,slots,64,True).cpu()
        factor=(1-.5**torch.arange(1,length+1))/(1-.5)*math.log(2)
        expected=(factor*128/(1+math.exp(-1)))[None,:,None].expand_as(actual)
        torch.testing.assert_close(actual,expected,atol=.0005,rtol=.00001)
        torch.testing.assert_close(state.cpu(),torch.full_like(state.cpu(),factor[-1].item()),atol=.00001,rtol=.00001)
        known.append({'length':length,'max_absolute':(actual-expected).abs().max().item()})
    for batch,length in ((1,17),(1,128),(1,512),(1,1536),(4,128)):
        heads,width,states=24,64,128;inner=heads*width
        x=torch.randn(batch,length,inner+2*states,device='cuda')
        gate=torch.randn(batch,length,inner,device='cuda');dt=torch.randn(batch,length,heads,device='cuda')
        bias=torch.linspace(-5,0,heads,device='cuda');a=-torch.rand(heads,device='cuda');d=torch.randn(heads,device='cuda')
        original=torch.randn(4,inner,states,device='cuda');slots=torch.tensor([3,0,2,1][:batch],device='cuda')
        for reset in (True,False):
            saved=[]
            for module in (mamba2_kernels,mamba2_ssd):
                state=original.clone();out=module.scan(x,gate,dt,bias,a,d,state,slots,width,reset)
                saved.append((out.clone(),state.clone()))
            for index in range(2):torch.testing.assert_close(saved[0][index],saved[1][index],atol=.003,rtol=.001)
            timings={'recurrent':[],'ssd':[]}
            for label,module in [('recurrent',mamba2_kernels),('ssd',mamba2_ssd),('ssd',mamba2_ssd),('recurrent',mamba2_kernels)]*3:
                state=original.clone()
                # Capture repeated dispatch so event timing measures device work.
                module.scan(x,gate,dt,bias,a,d,state,slots,width,reset);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):module.scan(x,gate,dt,bias,a,d,state,slots,width,reset)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):graph.replay()
                end.record();end.synchronize();timings[label].append(begin.elapsed_time(end)/10)
            reports.append({'batch':batch,'length':length,'reset':reset,
                'uses_ssd':length>=128,
                'max_output_error':(saved[0][0]-saved[1][0]).abs().max().item(),
                'max_state_error':(saved[0][1]-saved[1][1]).abs().max().item(),
                'median_ms':{k:statistics.median(v) for k,v in timings.items()},'trials_ms':timings})
    print(json.dumps({'passed':True,'device':'cuda','checks':reports,'closed_form_checks':known},indent=2))


if __name__=='__main__':main()
