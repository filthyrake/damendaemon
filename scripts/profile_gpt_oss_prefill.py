"""Attribute warm native GPT-OSS prefill work; profiling is not a latency benchmark."""
import argparse
import json
import os
from pathlib import Path
import torch
from dllm.qwen2 import ModelWeights
from dllm.gpt_oss import GptOssExecutor


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--tokens',type=int,default=4000)
    p.add_argument('--output',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    if args.tokens<1:raise ValueError('positive prompt length required')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    context=1<<(args.tokens-1).bit_length();model=ModelWeights(args.model,context)
    executor=GptOssExecutor(model,max_batch=2)
    base=model.raw_tokens('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ')
    ids=(base*((args.tokens+len(base)-1)//len(base)))[:args.tokens]
    executor.prefill(ids);torch.cuda.synchronize()
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.profiler.record_function('dllm.prefill'):executor.prefill(ids)
        torch.cuda.synchronize()
    rows=[{'operator':e.key,'device_type':str(e.device_type),'calls':e.count,'self_cpu_us':e.self_cpu_time_total,
        'cpu_total_us':e.cpu_time_total,'self_device_us':e.self_device_time_total,
        'device_total_us':e.device_time_total} for e in profile.key_averages()]
    report={'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'model':args.model,
        'tokens':args.tokens,'context':context,'scope':'Operator attribution only; profiler overhead is not a latency measurement.',
        'top_device':sorted(rows,key=lambda r:r['self_device_us'],reverse=True)[:30],
        'top_host':sorted(rows,key=lambda r:r['self_cpu_us'],reverse=True)[:30],
        'rows':rows}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k][:10] for k in ('top_device','top_host')},indent=2))


if __name__=='__main__':main()
