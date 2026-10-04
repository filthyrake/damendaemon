"""T5 relative-bias GPU attention against independent CPU tensor attention."""
import argparse
import os
import json
import torch
from torch.nn import functional as F
from dllm.t5 import relative_buckets
from dllm.runtime import require_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=p.parse_args();device=args.device;os.environ['DLLM_DEVICE']=device
    if device=='cuda':from dllm.kernels import attention
    else:from dllm.metal_kernels import attention
    require_gpu();torch.manual_seed(29)
    checks=[]
    for dtype in ((torch.float16,torch.float32) if device=='cuda' else (torch.float32,)):
        for dim in (64,96,128):
            query=torch.randn((4,6,1,dim),device=device,dtype=dtype)
            keys=torch.randn((4,3,513,dim),device=device,dtype=dtype)
            values=torch.randn_like(keys)
            slots=torch.tensor([3,0,2,1],device=device)
            positions=torch.tensor([15,16,128,512],device=device)
            bias=torch.randn((6,32),device=device)
            expected=[]
            for row,(slot,end) in enumerate(zip(slots.tolist(),positions.tolist())):
                buckets=relative_buckets(torch.tensor([end]),torch.arange(end+1))
                k=keys[slot:slot+1,:,:end+1].float().cpu().repeat_interleave(2,dim=1)
                v=values[slot:slot+1,:,:end+1].float().cpu().repeat_interleave(2,dim=1)
                expected.append(F.scaled_dot_product_attention(query[row:row+1].float().cpu(),k,v,
                    attn_mask=bias.cpu()[:,buckets].unsqueeze(0),scale=1.))
                keys[slot,:,end+1:]=float('nan');values[slot,:,end+1:]=float('nan')
            actual=attention(query,keys,values,positions,slots,scale=1.,relative_bias=bias)
            expected=torch.cat(expected).to(device=device,dtype=dtype)
            torch.testing.assert_close(actual,expected,atol=.004 if dtype==torch.float16 else .00005,rtol=.004)
            checks.append({'dtype':str(dtype),'dim':dim,'max_error':(actual-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name() if device=='cuda' else 'Apple M4','checks':checks},indent=2))


if __name__=='__main__':main()
