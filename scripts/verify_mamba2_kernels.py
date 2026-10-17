"""Mamba-2 native recurrence against independent CPU FP32 equations."""
import argparse
import json
import os
import torch
from torch.nn import functional as F
from damendaemon.runtime import require_gpu


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cuda','mps'),default='cuda');args=p.parse_args()
    device=args.device;os.environ['DAMENDAEMON_DEVICE']=device;require_gpu();torch.manual_seed(319)
    if device=='cuda':from damendaemon import mamba2_kernels as native
    else:from damendaemon import mamba2_metal as native
    checks=[]
    for heads,width,states in ((3,7,1),(2,16,33),(3,64,128),(2,32,256)):
        inner=heads*width;batch=3;slots=torch.tensor([3,0,2]);head=torch.arange(inner)//width
        state=torch.randn(4,inner,states);gpu_state=state.to(device)
        a=-torch.rand(heads);d=torch.randn(heads);bias=torch.linspace(-15,15,heads)
        for reset,length in ((True,3),(False,1),(False,17),(True,1)):
            x=torch.randn(batch,length,inner+2*states);gate=torch.randn(batch,length,inner);dt=torch.randn(batch,length,heads)
            expected=[]
            for row,slot in enumerate(slots.tolist()):
                if reset:state[slot].zero_()
                result=[]
                for t in range(length):
                    signal,b,c=x[row,t].split((inner,states,states))
                    step=F.softplus(dt[row,t]+bias)[head,None]
                    state[slot]=torch.exp(step*a[head,None])*state[slot]+(step*b[None,:])*signal[:,None]
                    result.append(((state[slot]*c[None,:]).sum(-1)+d[head]*signal)*F.silu(gate[row,t]))
                expected.append(torch.stack(result))
            actual=native.scan(x.to(device),gate.to(device),dt.to(device),bias.to(device),a.to(device),d.to(device),gpu_state,slots.to(device),width,reset).cpu()
            expected=torch.stack(expected)
            torch.testing.assert_close(actual,expected,atol=.003,rtol=.0005)
            torch.testing.assert_close(gpu_state.cpu(),state,atol=.003,rtol=.0005)
            checks.append({'heads':heads,'head_dim':width,'states':states,'reset':reset,'length':length,
                'max_absolute':(actual-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':device,'checks':checks},indent=2))


if __name__=='__main__':main()
