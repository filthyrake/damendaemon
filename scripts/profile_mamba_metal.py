"""Counterbalanced Metal time-step fusion component timing, not serving speed."""
import json
import time
import torch
from dllm import metal_kernels as linear
from dllm import mamba_metal as recurrent


@torch.inference_mode()
def main():
    torch.manual_seed(73);reports=[]
    for batch in (1,4):
        for inner,rank in ((1536,48),(2048,64)):
            state_size=16
            x=torch.randn(batch,1,inner,device='mps');gate=torch.randn_like(x)
            params=torch.randn(batch,1,rank+2*state_size,device='mps')
            weight=torch.randn(inner,rank,device='mps')*.1;bias=torch.randn(inner,device='mps')-4
            a=-torch.rand(inner,state_size,device='mps');d=torch.randn(inner,device='mps')
            state=torch.zeros(batch,inner,state_size,device='mps');slots=torch.arange(batch,device='mps')
            def operation(fused):
                dt=None if fused else linear.linear(params[:,:,:rank].contiguous().reshape(batch,rank),weight).reshape(batch,1,inner)
                return recurrent.scan(x,gate,params,dt,bias,a,d,state,slots,True,**({'dt_weight':weight} if fused else {}))
            torch.testing.assert_close(operation(True),operation(False),atol=.0001,rtol=.0001)
            for _ in range(5):operation(True);operation(False)
            rounds=[]
            for fused in (False,True,True,False)*3:
                torch.mps.synchronize();start=time.perf_counter()
                for _ in range(50):operation(fused)
                torch.mps.synchronize()
                rounds.append({'fused':fused,'microseconds':(time.perf_counter()-start)*1.e6/50})
            reports.append({'batch':batch,'inner':inner,'rank':rank,'rounds':rounds})
    print(json.dumps({'passed':True,'reports':reports,'scope':'projection plus recurrent update component; not full inference'},indent=2))


if __name__=='__main__':main()
