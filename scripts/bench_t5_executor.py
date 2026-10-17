"""Single-slot core inference timing, separate from HTTP serving measurements."""
import argparse
import json
import time
import torch
from damendaemon.models import load
from damendaemon.runtime import require_gpu,make_executor
from damendaemon.device import synchronize


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt');args=p.parse_args()
    require_gpu();torch.set_num_threads(4)
    executor=make_executor(load(args.model,512),max_batch=1)
    prompt=json.load(open(args.prompt));trials=[]
    for trial in range(6):
        synchronize();start=time.perf_counter()
        logits=executor.prefill(prompt);output=[]
        for step in range(8):
            token=logits.argmax(-1).item();output.append(token)
            if step==0:first=time.perf_counter()-start
            if token in executor.model.eos:break
            if step<7:logits,_=executor.decode(token)
        elapsed=time.perf_counter()-start
        trials.append({'warmup':trial==0,'first_token_seconds':first,'total_seconds':elapsed,'tokens':output})
    print(json.dumps({'engine':'damendaemon executor','context':512,'trials':trials},indent=2))


if __name__=='__main__':main()
