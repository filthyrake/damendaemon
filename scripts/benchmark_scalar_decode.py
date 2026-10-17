"""Matched old scalar launch path versus captured one-row decode on one GPU model."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
from damendaemon.qwen2 import ModelWeights,Executor
from damendaemon.runtime import make_executor


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    torch.set_num_threads(4)
    model=ModelWeights(args.model,512);executor=make_executor(model,max_batch=2)
    assert executor.graph is None and 1 in executor.batch_graphs
    base=model.raw_tokens('The capital of France is Paris. Explain how the ocean affects coastal weather. ')
    corpus=(base*300)[:512];rows=[]
    for length in (5,64,256):
        prompt=corpus[:length];decode_tokens=corpus[length:length+32]
        def run(graph,save=False):
            logits=executor.prefill(prompt);token=logits.argmax(-1).item();torch.cuda.synchronize()
            scores=[];generated=[];start=time.perf_counter()
            for step in range(32):
                supplied=decode_tokens[step] if save else token
                logits,next_id=executor.decode(supplied) if graph else Executor.decode(executor,supplied)
                token=next_id.item();generated.append(token)
                if save:scores.append(logits.clone())
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            return elapsed,generated,torch.cat(scores).cpu() if save else None
        old=run(False,True);new=run(True,True)
        delta=new[2].float()-old[2].float()
        rms=float(delta.square().mean().sqrt()/old[2].float().square().mean().sqrt().clamp_min(1e-9))
        maximum=float(delta.abs().max());assert rms<.01 and maximum<.5 and new[1]==old[1]
        # Graph metadata updates must not touch an independently populated slot.
        executor.prefill(corpus[:9],1);saved=[kv[:,1].clone() for kv in executor.kv]
        run(True);assert all(torch.equal(kv[:,1],value) for kv,value in zip(executor.kv,saved));del saved
        expected=run(False)[1];assert run(True)[1]==expected
        times=[]
        for trial in range(6):
            for graph in ((False,True) if trial%2==0 else (True,False)):
                elapsed,actual,_=run(graph);assert actual==expected
                times.append({'graph':graph,'seconds':elapsed})
        medians={str(graph):statistics.median(t['seconds'] for t in times if t['graph']==graph) for graph in (False,True)}
        row={'prompt_tokens':length,'decode_steps':32,'relative_rms':rms,'max_absolute_error':maximum,
            'teacher_forced_greedy_exact':True,'free_running_tokens_exact':True,'other_slot_unchanged':True,
            'median_decode_seconds':medians,'speedup':medians['False']/medians['True'],'measurements':times}
        rows.append(row);print(json.dumps(row),flush=True)
    result={'passed':True,'gpu':torch.cuda.get_device_name(),'model':args.model,'context':512,
        'compute_dtype':str(executor.compute_dtype),'residual_dtype':str(executor.residual_dtype),'weight_format':model.weight_format,
        'rows':rows,'method':'One resident checkpoint, fixed context and precision. 32 greedy decode steps after untimed prefill, six alternating warmed pairs, synchronization through next-token read as in direct runtime. Teacher-forced full-logit correctness measured separately. Excludes load, prefill, HTTP and chat parsing.'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
