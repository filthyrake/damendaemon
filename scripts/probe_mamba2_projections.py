"""Isolated compensated projection check, including exact weight conversion cost."""
import json,torch
from damendaemon.precise_matmul import linear
from torch.nn import functional as F
from triton.testing import do_bench_cudagraph
rows=[]
for m,n,k in [(128,3336,768),(1536,3336,768),(1536,768,1536)]:
 x=torch.randn(m,k,device='cuda');w=(torch.randn(n,k,device='cuda')*.03).half();wf=w.float()
 actual=linear(x,w);expected=F.linear(x,wf)
 error=((actual-expected).square().mean()/expected.square().mean()).sqrt().item()
 torch.testing.assert_close(actual,expected,atol=.001,rtol=.001)
 rows.append({'shape':[m,n,k],'relative_rms':error,'fp32_ms':do_bench_cudagraph(lambda:F.linear(x,wf),rep=100),'compensated_ms':do_bench_cudagraph(lambda:linear(x,w),rep=100),'with_weight_conversion_ms':do_bench_cudagraph(lambda:linear(x,wf.half()),rep=100)})
print(json.dumps(rows,indent=2))
