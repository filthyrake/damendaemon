"""Profile fixed/paged prompt and decode work; not a throughput benchmark."""
from dllm.evidence_environment import report_environment
import argparse
from collections import defaultdict
import json
import os
import torch
from dllm.runtime import require_gpu,make_executor
from dllm.models import load


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--prompt-length',type=int,default=32)
    p.add_argument('--batch',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.batch<=16 or not 1<=args.prompt_length<=args.context-4:p.error('invalid batch or prompt length')
    require_gpu()
    torch.set_num_threads(4)
    model=load(args.path,args.context)
    executor=make_executor(model,max_batch=args.batch)
    seed=model.raw_tokens('Explain how computers store information. ')
    prompts=[(seed*((args.prompt_length+len(seed)-1)//len(seed)))[:args.prompt_length] for _ in range(args.batch)]
    slots=list(range(args.batch))
    def prefill():return executor.prefill_batch(prompts,slots)
    prefill()
    def decode():return executor.decode_batch([10]*args.batch,slots)
    decode()
    reports={}
    for phase,operation in (('prefill',prefill),('decode',decode)):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            for _ in range(4):operation()
            torch.cuda.synchronize()
        groups=defaultdict(lambda:{'calls':0,'total_us':0})
        for event in profile.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                groups[event.name]['calls']+=1
                groups[event.name]['total_us']+=event.self_device_time_total
        reports[phase]={'steps':4,'gpu_us_per_step':sum(x['total_us'] for x in groups.values())/4,
            'kernels':sorted([{'name':name,**value} for name,value in groups.items()],key=lambda x:x['total_us'],reverse=True)}
    print(json.dumps({'layout':'paged' if hasattr(executor,'pool') else 'fixed','reports':reports,
        'model':args.path,'prompt_length':args.prompt_length,'batch':args.batch,
        'runtime_environment':report_environment(os.environ)},indent=2))


if __name__=='__main__':main()
