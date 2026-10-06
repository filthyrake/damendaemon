"""Native Metal Q8 projection against independently dequantized CPU FP64."""
import json
import torch
from torch.nn import functional as F
from dllm.quantized_metal import linear


@torch.inference_mode()
def main():
    torch.manual_seed(321);torch.set_num_threads(4);checks=[]
    for dtype in (torch.float32,torch.float16):
        for rows,outputs,width in ((1,7,32),(2,31,96),(3,65,256),(4,129,768),(5,33,64),(17,65,1536),(31,13,128)):
            for biased in (False,True):
                q=torch.randint(-128,128,(outputs,width),dtype=torch.int8)
                scales=((torch.rand(outputs,width//32)-.5)*.06).half();scales[::3,0]=0
                x=(torch.randn(rows,width)*(3 if dtype==torch.float16 else 1.e4)).to(dtype)
                if dtype==torch.float32:x[:,-1]=1.e5  # Preserve FP32 range.
                bias=torch.randn(outputs).to(dtype) if biased else None
                dequant=(q.double().reshape(outputs,width//32,32)*scales.double()[:,:,None]).reshape(outputs,width)
                if dtype==torch.float16:dequant=dequant.half().double()
                expected=F.linear(x.double(),dequant,None if bias is None else bias.double())
                actual=linear(x.to('mps'),q.to('mps'),scales.to('mps'),None if bias is None else bias.to('mps')).cpu().double()
                error=actual-expected;relative=(error.square().mean()/expected.square().mean().clamp_min(1.e-12)).sqrt().item()
                assert relative<(.0005 if dtype==torch.float16 else .00001),(rows,outputs,width,relative)
                assert torch.isfinite(actual).all()
                checks.append({'dtype':str(dtype),'rows':rows,'outputs':outputs,'width':width,'bias':biased,'relative_rms':relative,
                    'max_absolute':error.abs().max().item()})
    print(json.dumps({'passed':True,'device':'mps','checks':checks,'reference':'CPU FP64 direct signed integer times scale, with FP16 coefficient rounding for FP16 activations'},indent=2))


if __name__=='__main__':main()
