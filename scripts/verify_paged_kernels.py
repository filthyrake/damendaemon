"""Fragmented CUDA page addressing against a contiguous PyTorch attention oracle."""
import json
import torch
from torch.nn import functional as F
from dllm import kernels
from dllm.runtime import require_gpu


def main():
    require_gpu()
    torch.manual_seed(2026)
    checks=[]
    for dtype in (torch.float16,torch.float32):
        for dim in (64,96,128):
            for window in (0,33):
                heads,kh,context,page=8,2,513,16
                slots=torch.tensor([3,0,2],device='cuda')
                pos=torch.tensor([15,256,512],device='cuda')
                logical_k=torch.randn((4,kh,context,dim),device='cuda',dtype=dtype)
                logical_v=torch.randn_like(logical_k)
                table=torch.full((4,33),-1,device='cuda',dtype=torch.int32)
                physical_k=torch.full((80,kh,page,dim),float('nan'),device='cuda',dtype=dtype)
                physical_v=torch.full_like(physical_k,float('nan'))
                free=torch.randperm(80).tolist()
                for slot,position in zip(slots.tolist(),pos.tolist()):
                    for first in range(0,position+1,page):
                        block=free.pop()
                        table[slot,first//page]=block
                        count=min(page,position+1-first)
                        physical_k[block,:,:count]=logical_k[slot,:,first:first+count]
                        physical_v[block,:,:count]=logical_v[slot,:,first:first+count]
                packed=torch.randn((3,(heads+2*kh)*dim),device='cuda',dtype=dtype)
                angles=torch.randn((context,dim//2),device='cuda')
                angles=torch.cat((angles,angles),dim=-1)
                cos,sin=angles.cos(),angles.sin()
                # Use the precise path to compare against explicit FP32 rotation.
                query=kernels.rope_batch(packed,cos,sin,pos,slots,physical_k,physical_v,heads,
                                          precise=True,page_table=table)
                q,k,v=packed.split([heads*dim,kh*dim,kh*dim],-1)
                def rotate(x,h):
                    x=x.reshape(3,h,dim).float()
                    a,b=x.chunk(2,-1)
                    return (x*cos[pos,None]+torch.cat((-b,a),-1)*sin[pos,None]).to(dtype)
                expected_q=rotate(q,heads).unsqueeze(2)
                expected_k=rotate(k,kh)
                torch.testing.assert_close(query,expected_q,atol=.002 if dtype==torch.float16 else .000002,rtol=.002)
                direct_q,direct_k,direct_v=kernels.rope_batch(packed,cos,sin,pos,slots,physical_k,physical_v,heads,
                    precise=True,page_table=table,materialize=True)
                torch.testing.assert_close(direct_q,query,atol=0,rtol=0)
                torch.testing.assert_close(direct_k,expected_k,atol=.002 if dtype==torch.float16 else .000002,rtol=.002)
                torch.testing.assert_close(direct_v,v.reshape(3,kh,dim),atol=0,rtol=0)
                for row,(slot,position) in enumerate(zip(slots.tolist(),pos.tolist())):
                    block=table[slot,position//page].item()
                    torch.testing.assert_close(physical_k[block,:,position%page],expected_k[row],atol=.002,rtol=.002)
                    torch.testing.assert_close(physical_k[block,:,position%page],direct_k[row],atol=0,rtol=0)
                    torch.testing.assert_close(physical_v[block,:,position%page],v[row].reshape(kh,dim),atol=0,rtol=0)
                    logical_k[slot,:,position]=expected_k[row]
                    logical_v[slot,:,position]=v[row].reshape(kh,dim)
                actual=kernels.attention(query,physical_k,physical_v,pos,slots,window=window,page_table=table)
                expected=[]
                for row,(slot,position) in enumerate(zip(slots.tolist(),pos.tolist())):
                    begin=max(0,position-window+1) if window else 0
                    keys=logical_k[slot:slot+1,:,begin:position+1].float().repeat_interleave(heads//kh,dim=1)
                    values=logical_v[slot:slot+1,:,begin:position+1].float().repeat_interleave(heads//kh,dim=1)
                    expected.append(F.scaled_dot_product_attention(expected_q[row:row+1].float(),keys,values))
                expected=torch.cat(expected).to(dtype)
                torch.testing.assert_close(actual,expected,atol=.003 if dtype==torch.float16 else .00002,rtol=.003)
                checks.append({'dtype':str(dtype),'dim':dim,'window':window,'max_error':(actual-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'checks':checks},indent=2))


if __name__=='__main__':main()
