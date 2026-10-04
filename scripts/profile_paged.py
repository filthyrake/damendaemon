"""Profile fixed/paged prompt and decode work; not a throughput benchmark."""
import argparse
from collections import defaultdict
import json
import torch
from dllm.runtime import require_gpu,make_executor
from dllm.qwen2 import ModelWeights


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    args=p.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    model=ModelWeights(args.path,2048)
    executor=make_executor(model,max_batch=4)
    seed=model.raw_tokens('Explain how computers store information. ')
    prompts=[(seed*20)[:32] for _ in range(4)]
    slots=list(range(4))
    def prefill():return executor.prefill_batch(prompts,slots)
    prefill()
    def decode():return executor.decode_batch([10]*4,slots)
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
    print(json.dumps({'layout':'paged' if hasattr(executor,'pool') else 'fixed','reports':reports},indent=2))


if __name__=='__main__':main()
