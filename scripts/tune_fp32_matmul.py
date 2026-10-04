"""Negative-result A2 projection tile sweep; not used by the runtime."""
import torch,triton,json,itertools
from fp32_matmul_experiment import _matmul
from torch.nn import functional as F
from triton.testing import do_bench_cudagraph
results=[]
for m,n,k in [(128,3336,768),(1536,3336,768),(1536,768,1536)]:
 x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda');y=torch.empty(m,n,device='cuda');expected=F.linear(x,w)
 values=[]
 for bm,bn,bk,warps,stages in itertools.product([32,64],[64,128],[32,64],[4,8],[1,3]):
  def op():return _matmul[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,w,x,y,m,n,k,bm,bn,bk,False,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
  try:
   op();torch.testing.assert_close(y,expected,atol=.001,rtol=.001)
   ms=do_bench_cudagraph(op,rep=60)
   values.append(dict(config=[bm,bn,bk,warps,stages],ms=ms))
  except Exception as e:values.append(dict(config=[bm,bn,bk,warps,stages],error=str(e)[:300]))
 results.append(dict(shape=[m,n,k],cublas_ms=do_bench_cudagraph(lambda:F.linear(x,w),rep=60),candidates=values))
 print(json.dumps(results[-1]),flush=True)
