"""Counterbalanced in-process Mamba decode timing for the fusion policy."""
import argparse
import json
import time
import torch
from dllm.models import load
from dllm.runtime import make_executor,require_gpu


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    require_gpu();torch.set_num_threads(4)
    model=load(args.model,512);executor=make_executor(model,max_batch=4)
    prompt=model.raw_tokens('The blue bird flies over the green field every morning. It')
    reports=[]
    for batch in (1,4):
        slots=list(range(batch));rounds=[];expected=None
        for fused in (False,True):
            executor.fuse_time_step=fused
            tokens=executor.prefill_batch([prompt]*batch,slots).argmax(-1).cpu().tolist()
            for _ in range(8):
                _,selected=executor.decode_batch(tokens,slots);tokens=selected.cpu().tolist()
        torch.mps.synchronize()
        for index,fused in enumerate((False,True,True,False)*2):
            executor.fuse_time_step=fused
            tokens=executor.prefill_batch([prompt]*batch,slots).argmax(-1).cpu().tolist()
            torch.mps.synchronize();start=time.perf_counter();sequence=[]
            for _ in range(32):
                _,selected=executor.decode_batch(tokens,slots);tokens=selected.cpu().tolist();sequence.append(tokens)
            elapsed=time.perf_counter()-start
            if expected is None:expected=sequence
            assert sequence==expected
            rounds.append({'fused':fused,'seconds':elapsed,'tokens_per_second':32*batch/elapsed})
        reports.append({'batch':batch,'rounds':rounds})
    print(json.dumps({'passed':True,'reports':reports,'scope':'warm core decode only; same live executor and token sequence'},indent=2))


if __name__=='__main__':main()
