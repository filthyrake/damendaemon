"""Bounded attention-allocation diagnosis for #73; no model weights or tuning."""
import gc
import json
import math
import os

import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right

from dllm.device import attention
from dllm.rocm_policy import host_available, memory_limit
from dllm.runtime import require_gpu


@torch.inference_mode()
def main():
    if os.environ.get('DLLM_DEVICE') != 'rocm' or memory_limit() != 2*1024**3:
        raise ValueError('requires explicit ROCm and a 2-GiB allocator cap')
    if host_available() < 32*1024**3:
        raise MemoryError('diagnostic requires 32 GiB available host RAM')
    require_gpu()
    torch.manual_seed(73)
    rows = []
    for length in (1024, 4096, 8192, 16384, 32768):
        q = torch.randn(1, 24, 127, 256, device='cuda')
        k = torch.randn(1, 4, length, 256, device='cuda')
        v = torch.randn_like(k)
        # Direct single-query attention checks do not use SDPA or GQA dispatch.
        samples = [(h, i) for h in (0, 5, 6, 23) for i in (0, 63, 126)]
        oracle = []
        for head, index in samples:
            end = length-127+index+1
            scores = q[0, head, index] @ k[0, head//6, :end].T / math.sqrt(256)
            oracle.append(scores.softmax(-1) @ v[0, head//6, :end])
        target = torch.stack(oracle)
        for route in ('current_gqa', 'expanded_kv_control'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = dict(kv_tokens=length, query_tokens=127, route=route,
                       host_available_before=host_available(), completed=False)
            try:
                bias = causal_lower_right(127, length)
                if route == 'current_gqa':
                    output = attention(q, k, v, attn_mask=bias, enable_gqa=True)
                else:
                    output = F.scaled_dot_product_attention(q, k.repeat_interleave(6, dim=1),
                        v.repeat_interleave(6, dim=1), attn_mask=bias)
                torch.cuda.synchronize()
                selected = torch.stack([output[0, head, index] for head, index in samples])
                difference = selected-target
                absolute = difference.abs().max().item()
                rms = (difference.square().mean().sqrt()/target.square().mean().sqrt()).item()
                row.update(completed=True, max_absolute=absolute, relative_rms=rms,
                           numerical_pass=absolute < 1e-4 and rms < 1e-4)
                del output, selected, difference
            except torch.OutOfMemoryError as error:
                row['allocator_oom'] = str(error)
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       host_available_after=host_available())
            rows.append(row)
            print(json.dumps(row), flush=True)
        del q, k, v, oracle, target, scores
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(dict(complete=True, rows=rows, torch=str(torch.__version__),
        device=torch.cuda.get_device_name(), budget_bytes=memory_limit(),
        limitation='Attention-only allocation diagnosis. No full-model attribution, performance or 32K qualification.')), flush=True)


if __name__ == '__main__':
    main()
