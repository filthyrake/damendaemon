"""T5 relative-bias CUDA attention against independent tensor attention."""
import json
import torch
from torch.nn import functional as F
from dllm.kernels import attention
from dllm.t5 import relative_buckets
from dllm.runtime import require_gpu


def main():
    require_gpu();torch.manual_seed(29)
    checks=[]
    for dtype in (torch.float16,torch.float32):
        for dim in (64,96,128):
            query=torch.randn((4,6,1,dim),device='cuda',dtype=dtype)
            keys=torch.randn((4,3,513,dim),device='cuda',dtype=dtype)
            values=torch.randn_like(keys)
            slots=torch.tensor([3,0,2,1],device='cuda')
            positions=torch.tensor([15,16,128,512],device='cuda')
            bias=torch.randn((6,32),device='cuda')
            expected=[]
            for row,(slot,end) in enumerate(zip(slots.tolist(),positions.tolist())):
                buckets=relative_buckets(torch.tensor([end]),torch.arange(end+1)).to('cuda')
                k=keys[slot:slot+1,:,:end+1].float().repeat_interleave(2,dim=1)
                v=values[slot:slot+1,:,:end+1].float().repeat_interleave(2,dim=1)
                expected.append(F.scaled_dot_product_attention(query[row:row+1].float(),k,v,
                    attn_mask=bias[:,buckets].unsqueeze(0),scale=1.))
                keys[slot,:,end+1:]=float('nan');values[slot,:,end+1:]=float('nan')
            actual=attention(query,keys,values,positions,slots,scale=1.,relative_bias=bias)
            expected=torch.cat(expected).to(dtype)
            torch.testing.assert_close(actual,expected,atol=.004 if dtype==torch.float16 else .00005,rtol=.004)
            checks.append({'dtype':str(dtype),'dim':dim,'max_error':(actual-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'checks':checks},indent=2))


if __name__=='__main__':main()
