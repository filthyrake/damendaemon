"""Independent CUDA matrix-vector gate, including values outside FP16 range."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import torch
from dllm.runtime import require_gpu
from dllm.kernels import linear_fp32

require_gpu()
torch.manual_seed(17)
checks = []
with torch.inference_mode():
    for rows, width, outputs in ((1,640,262144),(2,2048,640),(4,1024,640),(3,73,117),(7,73,117)):
        for dtype in (torch.float16, torch.float32):
            x = torch.randn(rows,width,device='cuda',dtype=torch.float32) * 100000
            w = torch.randn(outputs,width,device='cuda',dtype=dtype) * .01
            expected = torch.nn.functional.linear(x,w.float())
            actual = linear_fp32(x,w)
            relative = ((actual-expected).square().mean()/expected.square().mean()).sqrt().item()
            passed = torch.isfinite(actual).all().item() and relative < 1e-5
            checks.append({'rows':rows,'width':width,'outputs':outputs,'weight_dtype':str(dtype),
                           'relative_rms':relative,'passed':passed})
print(json.dumps({'checks':checks,'passed':all(x['passed'] for x in checks)},indent=2))
assert all(x['passed'] for x in checks)
