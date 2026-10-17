"""Numerical gate for compensated products, including scaled activation ranges."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import torch
from damendaemon.precise_matmul import linear
from damendaemon.runtime import require_gpu

require_gpu()
torch.manual_seed(13)
checks=[]
for rows,width,outputs in ((2,96,113),(3,1024,4096),(4,3072,1024),(17,128,71),
                           (26,1024,6144),(65,96,113),(128,96,113)):
    w=(torch.randn(outputs,width,device='cuda')*.02).half()
    for scale in (0.,1.e-8,1.,1.e6,1.e20):
        x=torch.randn(rows,width,device='cuda')*scale
        expected=torch.nn.functional.linear(x,w.float())
        actual=linear(x,w)
        delta=(actual.double()-expected.double()).square().mean().sqrt()
        relative=(delta/expected.double().square().mean().sqrt().clamp_min(1.e-30)).item()
        assert torch.isfinite(actual).all() and relative<1.e-5,(rows,width,scale,relative)
        checks.append({'rows':rows,'width':width,'outputs':outputs,'scale':scale,'relative_rms':relative})
print(json.dumps({'passed':True,'checks':checks},indent=2))
