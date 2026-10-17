"""Real A2 FP32 GQA accuracy, kernel selection, latency and peak allocation."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import json
import os
import statistics
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from damendaemon.device import attention


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);torch.manual_seed(42);rows=[]
    def measure(call):
        for _ in range(2):call()
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(5):
            start=time.perf_counter();call();torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
        return {'milliseconds':times,'median_ms':statistics.median(times),
                'peak_extra_allocated_bytes':torch.cuda.max_memory_allocated()-base}
    with torch.inference_mode():
        for length,offset,window in ((127,0,0),(257,0,0),(512,0,0),(2048,0,0),(8192,0,0),(257,17,0),(257,17,127)):
            keys=length+offset
            q=torch.randn((1,32,length,64),device='cuda',dtype=torch.float32)
            k=torch.randn((1,8,keys,64),device='cuda',dtype=torch.float32);v=torch.randn_like(k)
            options={'enable_gqa':True}
            mask=None
            if offset or window:
                qp=torch.arange(offset,offset+length,device='cuda')[:,None]
                kp=torch.arange(keys,device='cuda')[None,:]
                mask=kp<=qp
                if window:mask=mask & (kp>qp-window)
                options['attn_mask']=mask
            else:options['is_causal']=True
            def native():return attention(q,k,v,**options)
            def baseline():
                with sdpa_kernel(SDPBackend.MATH):return F.scaled_dot_product_attention(q,k,v,**options)
            native_measurement=measure(native)
            baseline_measurement=measure(baseline) if length<=2048 else None
            out=native()
            ids=torch.tensor([0,1,min(127,length-1),length-2,length-1],device='cuda')
            selected_mask=(torch.arange(keys,device='cuda')[None,:]<=ids[:,None]) if mask is None else mask[ids]
            with sdpa_kernel(SDPBackend.MATH):
                target=F.scaled_dot_product_attention(q[:,:,ids],k,v,attn_mask=selected_mask,enable_gqa=True)
            error=(out[:,:,ids]-target).abs();maximum=error.max().item()
            torch.testing.assert_close(out[:,:,ids],target,rtol=2e-5,atol=3e-6)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:native()
            kernels=[event.key for event in profile.key_averages() if 'scaled_dot_product' in event.key]
            if length>=256 and not offset:
                assert any('efficient_attention' in name for name in kernels),kernels
            rows.append({'query_tokens':length,'key_tokens':keys,'query_start':offset,'window':window,
                'max_absolute_error':maximum,'rms_error':error.square().mean().sqrt().item(),
                'native':native_measurement,'math_baseline':baseline_measurement,'operators':kernels})
    print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'dtype':'float32','query_heads':32,'kv_heads':8,'head_dim':64,'rows':rows,
        'reference':'Independent math SDPA at selected first/boundary/last query rows; no 8192 full math allocation'},indent=2))


if __name__=='__main__':main()
