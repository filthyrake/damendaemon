"""Independent ALiBi slope and slotted CUDA attention validation."""
import json
import torch
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from dllm.bloom import alibi_slopes
from dllm import kernels
from dllm.runtime import require_gpu


def main():
    require_gpu()
    torch.manual_seed(2026)
    checks = []
    for heads in (1, 12, 16):
        slope = alibi_slopes(heads, 'cuda')
        reference = build_alibi_tensor(torch.ones((1, 513),device='cuda'), heads, torch.float32)
        torch.testing.assert_close(slope[:,None]*torch.arange(513,device='cuda'),reference[:,0])
        for dim in (64, 96, 128):
            for dtype in (torch.float16, torch.float32):
                slots = torch.tensor([3,0,2],device='cuda')
                positions = torch.tensor([0,256,512],device='cuda')
                k = torch.randn((4,heads,513,dim),device='cuda',dtype=dtype)
                v = torch.randn_like(k)
                q = torch.randn((3,heads,1,dim),device='cuda',dtype=dtype)
                for slot,pos in zip(slots.tolist(),positions.tolist()):
                    k[slot,:,pos+1:] = float('nan')
                    v[slot,:,pos+1:] = float('nan')
                actual = kernels.attention(q,k,v,positions,slots,alibi=slope)
                expected=[]
                for row,(slot,pos) in enumerate(zip(slots.tolist(),positions.tolist())):
                    scores = q[row].float() @ k[slot,:,:pos+1].float().transpose(-1,-2) / dim**.5
                    scores += reference[:,:,:pos+1]
                    expected.append(scores.softmax(-1) @ v[slot,:,:pos+1].float())
                expected=torch.stack(expected).to(dtype)
                torch.testing.assert_close(actual,expected,atol=.003 if dtype==torch.float16 else .00004,rtol=.003)
                checks.append({'heads':heads,'dim':dim,'dtype':str(dtype),'max_error':(actual-expected).abs().max().item()})
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name(),'checks':checks},indent=2))


if __name__=='__main__':main()
