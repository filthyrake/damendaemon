"""A2 large-row compensated projection experiment; does not alter runtime policy."""
import json
import os
import statistics
import time

import torch
from torch.nn import functional as F
from dllm.precise_matmul import linear


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);torch.manual_seed(57);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    def measure(call):
        for _ in range(2):call()
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(5):
            start=time.perf_counter();call();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
        return {'median_ms':statistics.median(times),'milliseconds':times,
                'peak_extra_bytes':torch.cuda.max_memory_allocated()-base}
    with torch.inference_mode():
        for count in (129,512,2048):
            for inputs,outputs in ((2048,3072),(2048,16384),(8192,2048)):
                x=torch.randn(count,inputs,device='cuda')
                w=(torch.randn(outputs,inputs,device='cuda')*.02).half()
                target=F.linear(x,w.float());actual=linear(x,w)
                error=(actual-target).abs();maximum=error.max().item()
                relative=(error.square().mean().sqrt()/target.square().mean().sqrt()).item()
                torch.testing.assert_close(actual,target,rtol=1e-4,atol=3e-5)
                baseline=measure(lambda:F.linear(x,w.float()))
                candidate=measure(lambda:linear(x,w))
                rows.append({'rows':count,'inputs':inputs,'outputs':outputs,'max_absolute_error':maximum,
                    'relative_rms':relative,'baseline':baseline,'candidate':candidate})
    print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'rows':rows,'method':'FP16 stored weights, FP32 inputs/output. Baseline includes converting weights to FP32 as current runtime does. Five synchronized trials after two warmups. Compensated arithmetic is an approximation, not bitwise FP32.'},indent=2))


if __name__=='__main__':main()
