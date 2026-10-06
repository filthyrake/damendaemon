"""Isolated gfx1151 attention backend proof; no runtime default change.

Force math and Flash SDPA on identical FP16 inputs with actual model geometry.
Compare both against CPU FP32 math, record selected operators and ABBA timings.
Primary context: https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_cuda.py
The upstream experimental gate motivates a probe, not a support/performance claim.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from dllm.config import decoder_config
from dllm.runtime import require_gpu
from dllm.device import selected_accelerator


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    if selected_accelerator() != 'rocm' or not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1151'):
        raise ValueError('isolated HOMER experiment only')
    if os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL') != '1':
        raise ValueError('set the experimental flag explicitly for this process before importing torch')
    torch.set_num_threads(4)
    c = decoder_config(json.loads((Path(args.model)/'config.json').read_text()))
    heads, kv_heads, dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    report = dict(passed=False, model=args.model, geometry=dict(heads=heads, kv_heads=kv_heads, head_dim=dim),
        device=torch.cuda.get_device_name(), torch=str(torch.__version__), hip=torch.version.hip,
        experimental_flag=os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'], cases=[],
        limitation='Synthetic attention only; model numerical, serving and lifecycle gates are separate.')
    generator = torch.Generator().manual_seed(73)
    try:
        prepared = []
        for length, total in ((512,512), (512,32768), (256,32768), (16,32768)):
            q = torch.randn(1,heads,length,dim,generator=generator,dtype=torch.float32).half()
            k = torch.randn(1,kv_heads,total,dim,generator=generator,dtype=torch.float32).half()
            v = torch.randn(1,kv_heads,total,dim,generator=generator,dtype=torch.float32).half()
            mask = torch.arange(total)[None,:] <= torch.arange(total-length,total)[:,None]
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask, enable_gqa=True)
            prepared.append((length,total,q.cuda(),k.cuda(),v.cuda(),reference))
        for length,total,q,k,v,reference in prepared:
            row = dict(query_tokens=length,cache_tokens=total,backends={})
            report['cases'].append(row)
            bias = causal_lower_right(length,total)
            def run(backend):
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q,k,v,attn_mask=bias,enable_gqa=True)
            for name,backend in (('math',SDPBackend.MATH),('flash',SDPBackend.FLASH_ATTENTION)):
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    actual = run(backend)
                    torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()-base
                operators = sorted({event.key for event in profile.key_averages() if 'scaled_dot_product' in event.key})
                value = actual.float().cpu()
                delta = value-reference
                relative = (delta.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-6)).item()
                absolute = delta.abs().max().item()
                passed = bool(torch.isfinite(value).all()) and relative < .001 and absolute < .01
                row['backends'][name] = dict(passed=passed,relative_rms=relative,max_absolute=absolute,
                    selected_operators=operators,additional_peak_bytes=peak,timing_ms=[])
                assert passed, f'{name} numerical check failed'
                assert any(name in key for key in operators), f'{name} backend selection not observed'
                del actual
                for _ in range(3): run(backend)
                torch.cuda.synchronize()
            for _ in range(5):
                for name,backend in (('math',SDPBackend.MATH),('flash',SDPBackend.FLASH_ATTENTION),
                                     ('flash',SDPBackend.FLASH_ATTENTION),('math',SDPBackend.MATH)):
                    begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10): run(backend)
                    end.record();end.synchronize()
                    row['backends'][name]['timing_ms'].append(begin.elapsed_time(end)/10)
            for result in row['backends'].values():
                result['median_ms'] = statistics.median(result['timing_ms'])
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
