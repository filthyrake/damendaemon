"""Bounded #68 dispatch comparison; no model weights or tuning sweep."""
import argparse
import json
import math
import os
import time
import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from dllm.device import attention
from dllm.rocm_policy import memory_limit
from dllm.runtime import require_gpu


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',choices=('expanded','grouped','streaming','matmul'),default='streaming')
    comparison=parser.parse_args().comparison
    routes={'expanded':('current','expanded_kv'), 'grouped':('current','expanded_kv','grouped_views'),
            'streaming':('current','streaming'), 'matmul':('current','grouped_matmul')}[comparison]
    order={'expanded':('current','expanded_kv','expanded_kv','current'),
           'grouped':('current','grouped_views','grouped_views','current','expanded_kv','grouped_views','grouped_views','expanded_kv'),
           'streaming':('current','streaming','streaming','current'),
           'matmul':('current','grouped_matmul','grouped_matmul','current')}[comparison]
    if os.environ.get('DLLM_DEVICE') != 'rocm' or memory_limit() != 2*1024**3:
        raise ValueError('requires ROCm and 2-GiB allocator cap')
    require_gpu()
    torch.manual_seed(68)
    for length in (4096, 16384, 32768):
        q=torch.randn(1,24,127,256,device='cuda')
        k=torch.randn(1,4,length,256,device='cuda');v=torch.randn_like(k)
        bias=causal_lower_right(127,length)
        indices=[(h,t) for h in (0,5,6,23) for t in (0,63,126)]
        expected=[]
        for h,t in indices:
            end=length-127+t+1
            scores=q[0,h,t] @ k[0,h//6,:end].T / math.sqrt(256)
            expected.append(scores.softmax(-1) @ v[0,h//6,:end])
        expected=torch.stack(expected)
        def run(route):
            if route=='current':return attention(q,k,v,attn_mask=bias,enable_gqa=True)
            if route=='grouped_matmul':
                from scripts.experiments.grouped_prefill_attention import matmul_attention
                return matmul_attention(q,k,v)
            if route=='streaming':
                from scripts.experiments.streaming_hybrid_attention import attention as streaming
                return streaming(q,k,v)
            if route=='grouped_views':
                from scripts.experiments.grouped_prefill_attention import attention as grouped
                return grouped(q,k,v,attn_mask=bias,enable_gqa=True)
            return F.scaled_dot_product_attention(q,k.repeat_interleave(6,1),v.repeat_interleave(6,1),attn_mask=bias)
        available=[]
        for route in routes:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            row=dict(event='numerical',route=route,kv_tokens=length,query_tokens=127)
            try:
                value=run(route);torch.cuda.synchronize()
                delta=torch.stack([value[0,h,t] for h,t in indices])-expected
                row.update(max_absolute=delta.abs().max().item(),relative_rms=(delta.square().mean()/expected.square().mean()).sqrt().item())
                row['passed']=row['max_absolute']<1e-4 and row['relative_rms']<1e-4
                del value,delta
                if not row['passed']:raise AssertionError('numerical failure')
                available.append(route)
            except torch.OutOfMemoryError:
                row.update(passed=False,allocator_oom=True)
            row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            print(json.dumps(row),flush=True)
        for round_index in range(2):
            for route in order:
                if route not in available:continue
                torch.cuda.synchronize();start=time.perf_counter()
                value=run(route);torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                del value
                print(json.dumps(dict(event='timing',route=route,kv_tokens=length,round=round_index,seconds=elapsed)),flush=True)
        del q,k,v,expected,scores
        torch.cuda.empty_cache()


if __name__=='__main__':main()
