"""Native Metal Q8 projection against independently dequantized CPU FP64."""
import json
import torch
from torch.nn import functional as F
from dllm.quantized_metal import linear


@torch.inference_mode()
def main():
    torch.manual_seed(321);torch.set_num_threads(4);checks=[]
    for rows,outputs,width in ((1,7,32),(2,31,96),(3,65,256),(4,129,768),(5,33,64),(17,65,1536),(31,13,128)):
        for biased in (False,True):
            q=torch.randint(-128,128,(outputs,width),dtype=torch.int8)
            scales=(torch.rand(outputs,width//32)*.03).half();scales[:,0]=0
            x=torch.randn(rows,width)*1.e4
            x[:,-1]=1.e5  # Activations must not silently round/saturate to FP16.
            bias=torch.randn(outputs) if biased else None
            dequant=(q.double().reshape(outputs,width//32,32)*scales.double()[:,:,None]).reshape(outputs,width)
            expected=F.linear(x.double(),dequant,None if bias is None else bias.double())
            actual=linear(x.to('mps'),q.to('mps'),scales.to('mps'),None if bias is None else bias.to('mps')).cpu().double()
            error=actual-expected;relative=(error.square().mean()/expected.square().mean().clamp_min(1.e-12)).sqrt().item()
            assert relative<.00001,(rows,outputs,width,relative)
            assert torch.isfinite(actual).all()
            checks.append({'rows':rows,'outputs':outputs,'width':width,'bias':biased,'relative_rms':relative,
                'max_absolute':error.abs().max().item()})
    print(json.dumps({'passed':True,'device':'mps','checks':checks,'reference':'CPU FP64 direct signed integer times scale'},indent=2))


if __name__=='__main__':main()
