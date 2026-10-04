"""Independent ALiBi slope and slotted CUDA attention validation."""
import json
import argparse
import torch
from torch.nn import functional as F
from transformers.models.bloom.modeling_bloom import build_alibi_tensor
from dllm.bloom import alibi_slopes
from dllm.runtime import require_gpu


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=parser.parse_args()
    device=args.device
    require_gpu()
    if device=='cuda':
        from dllm import kernels
    else:
        from dllm import metal_kernels as kernels
    torch.manual_seed(2026)
    checks = []
    for heads in (1, 12, 16):
        slope = alibi_slopes(heads, device)
        reference = build_alibi_tensor(torch.ones((1, 513),device=device), heads, torch.float32)
        torch.testing.assert_close(slope[:,None]*torch.arange(513,device=device),reference[:,0])
        for dim in (64, 96, 128):
            for dtype in ((torch.float16, torch.float32) if device=='cuda' else (torch.float32,)):
                slots = torch.tensor([3,0,2],device=device)
                positions = torch.tensor([0,256,512],device=device)
                k = torch.randn((4,heads,513,dim),device=device,dtype=dtype)
                v = torch.randn_like(k)
                q = torch.randn((3,heads,1,dim),device=device,dtype=dtype)
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
    print(json.dumps({'passed':True,'device':torch.cuda.get_device_name() if device=='cuda' else 'Apple MPS','checks':checks},indent=2))


if __name__=='__main__':main()
