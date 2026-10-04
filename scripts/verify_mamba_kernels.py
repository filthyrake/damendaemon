"""Native Mamba kernels against independent CPU convolution/recurrence."""
import argparse
import json
import os
import torch
from torch.nn import functional as F
from dllm.runtime import require_gpu


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cuda','mps'),default='cuda');args=p.parse_args()
    device=args.device;os.environ['DLLM_DEVICE']=device;require_gpu();torch.manual_seed(307)
    if device=='cuda':from dllm import mamba_kernels as native
    else:from dllm import mamba_metal as native
    checks=[]
    for inner,state_size,width,length in ((31,1,1,1),(128,16,4,3),(65,32,4,17),(32,16,8,31)):
        slots=torch.tensor([3,0,2]);batch=3;rank=5
        history=torch.randn(4,inner,width);state=torch.randn(4,inner,state_size)
        weight=torch.randn(inner,width);bias=torch.randn(inner)
        a=-torch.rand(inner,state_size);d=torch.randn(inner);dt_bias=torch.linspace(-15,15,inner)
        gh=history.to(device);gs=state.to(device)
        for reset,steps in ((True,length),(False,1),(False,1)):
            x=torch.randn(batch,steps,inner);gate=torch.randn_like(x)
            params=torch.randn(batch,steps,rank+2*state_size);dt=torch.randn_like(x)
            dt_weight=torch.randn(inner,rank)
            if device=='mps' and steps==1:dt=F.linear(params[:,:,:rank],dt_weight)
            expected_conv=[];expected_scan=[]
            for row,slot in enumerate(slots.tolist()):
                if reset:
                    history[slot]=F.pad(x[row].T,(width-steps,0))
                    conv=F.silu(F.conv1d(x[row].T[None],weight[:,None],bias,padding=width-1,groups=inner)[0,:,:steps].T)
                    state[slot].zero_()
                else:
                    history[slot]=history[slot].roll(-1,-1);history[slot,:,-1]=x[row,0]
                    conv=F.silu((history[slot]*weight).sum(-1)+bias)[None]
                expected_conv.append(conv)
                output=[]
                for t in range(steps):
                    step=F.softplus(dt[row,t]+dt_bias)[:,None]
                    b=params[row,t,rank:rank+state_size];c=params[row,t,rank+state_size:]
                    state[slot]=torch.exp(step*a)*state[slot]+(step*b)*conv[t,:,None]
                    output.append(((state[slot]*c).sum(-1)+d*conv[t])*F.silu(gate[row,t]))
                expected_scan.append(torch.stack(output))
            packed_history,packed_state=gh.clone(),gs.clone()
            actual_conv=native.conv(x.to(device),weight.to(device),bias.to(device),gh,slots.to(device),reset)
            actual=native.scan(actual_conv,gate.to(device),params.to(device),dt.to(device),dt_bias.to(device),a.to(device),d.to(device),gs,slots.to(device),reset)
            expected=torch.stack(expected_scan)
            torch.testing.assert_close(actual_conv.cpu(),torch.stack(expected_conv),atol=.00003,rtol=.00003)
            torch.testing.assert_close(gh.cpu(),history,atol=0,rtol=0)
            torch.testing.assert_close(actual.cpu(),expected,atol=.003,rtol=.0005)
            torch.testing.assert_close(gs.cpu(),state,atol=.003,rtol=.0005)
            packed=torch.cat((x,gate),-1).to(device)
            pc=native.conv(packed,weight.to(device),bias.to(device),packed_history,slots.to(device),reset)
            fused=device=='mps' and steps==1
            ps=native.scan(pc,packed,params.to(device),None if fused else dt.to(device),dt_bias.to(device),
                a.to(device),d.to(device),packed_state,slots.to(device),reset,
                **({'dt_weight':dt_weight.to(device)} if fused else {}))
            torch.testing.assert_close(ps.cpu(),expected,atol=.003,rtol=.0005)
            torch.testing.assert_close(packed_state.cpu(),state,atol=.003,rtol=.0005)
            torch.testing.assert_close(packed_history.cpu(),history,atol=0,rtol=0)
            checks.append({'inner':inner,'state':state_size,'conv':width,'length':steps,'reset':reset,
                'max_output_error':(actual.cpu()-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':device,'checks':checks,'packed_projection':True,'fused_time_step':device=='mps'},indent=2))


if __name__=='__main__':main()
