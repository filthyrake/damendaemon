"""Isolate score computation/readback cost; not an inference benchmark."""
import json
import os
import time
import torch
from damendaemon.logprobs import score_rows
from damendaemon.runtime import require_gpu


def main():
    require_gpu();torch.set_num_threads(4)
    rows=[]
    for batch in (1,4):
        for top in (0,5,20):
            logits=torch.randn(batch,151936,device='cuda',dtype=torch.float16)
            selected=logits.argmax(-1)
            for _ in range(5):score_rows(logits,selected,[top]*batch)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(100):score_rows(logits,selected,[top]*batch)
            seconds=(time.perf_counter()-start)/100
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                for _ in range(4):score_rows(logits,selected,[top]*batch)
            events=[event for event in profile.events() if event.device_type==torch.autograd.DeviceType.CUDA]
            names={}
            for event in events:
                value=names.setdefault(event.name,{'calls':0,'gpu_us':0})
                value['calls']+=1;value['gpu_us']+=event.self_device_time_total
            rows.append({'batch':batch,'top':top,'wall_ms_per_batch':seconds*1000,
                'gpu_events_per_batch':len(events)/4,'kernels':names})
    print(json.dumps({'device':torch.cuda.get_device_name(),'backend_policy':os.environ.get('DAMENDAEMON_LOGPROBS_KERNELS','auto'),'vocabulary':151936,'dtype':'float16','measurements':rows},indent=2))


if __name__=='__main__':main()
