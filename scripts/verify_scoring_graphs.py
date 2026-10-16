"""Candidate-head graph isolation, eviction, and matched eager/replay timing."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import gc
import json
import os
import statistics
import time
import weakref
import torch
from dllm.runtime import Runtime


def measure(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();values=[]
    for _ in range(15):
        start=time.perf_counter();fn();torch.cuda.synchronize();values.append((time.perf_counter()-start)*1000)
    return statistics.median(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    os.environ['DLLM_PREFILL_GRAPHS']='3';torch.set_num_threads(4)
    runtime=Runtime({'a':args.model},context=256,batch_size=4)
    checks=[];timings=[]
    try:
        with runtime.session('a') as (executor,_,_):
            cache=executor.prefill_graph_cache;assert cache is not None
            for count,token,candidates in [(16,40,[0,1]),(16,100,[0,1]),(16,40,[1,0]),
                    (16,40,[17,33]),(16,40,[0,1,100]),(128,40,[0,1]),(129,40,[0,1])]:
                prompt=[token]*count
                executor.prefill_graph_cache=None
                reference=executor.prefill(prompt).float().index_select(-1,torch.tensor(candidates,device='cuda')).clone()
                executor.prefill_graph_cache=cache
                actual=executor.score_candidates(prompt,candidates).clone()
                torch.testing.assert_close(actual,reference,atol=.00005,rtol=.00005)
                assert len(cache.entries)<=3
                assert executor._candidate_head is None and executor._candidate_ids is None
                checks.append({'tokens':count,'prompt_id':token,'candidates':candidates,
                    'max_logit_error':(actual-reference).abs().max().item()})
            # Normal generation must still use the full head after scoring.
            full=executor.prefill([40]*16).float().clone()
            assert full.shape[-1]==executor.c['vocab_size']
            for count in (16,128):
                prompt=[40]*count
                executor.prefill_graph_cache=None
                eager=measure(lambda:executor.score_candidates(prompt,[0,1,100,1000]))
                executor.prefill_graph_cache=cache
                cached=measure(lambda:executor.score_candidates(prompt,[0,1,100,1000]))
                timings.append({'tokens':count,'candidates':4,'eager_ms':eager,'graph_ms':cached,'speedup':eager/cached})
            head_refs=[weakref.ref(entry[4]) for entry in cache.entries.values() if entry[4] is not None]
            assert head_refs
            for count in (17,18,19):executor.prefill([40]*count)
            gc.collect();assert all(ref() is None for ref in head_refs)
            # Repeat a narrow request after eviction; must recapture correct rows.
            actual=executor.score_candidates([40]*16,[1,0]).clone()
            torch.testing.assert_close(actual,full[:,[1,0]],atol=.00005,rtol=.00005)
            stats={'hits':cache.hits,'captures':cache.captures,'evictions':cache.evictions}
            del executor,cache
        runtime.manage('unload',model='a')
        print(json.dumps({'passed':True,'model':args.model,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'checks':checks,'timings':timings,'cache':stats,
            'eviction_frees_candidate_heads':True,'normal_generation_full_vocabulary':True,
            'timing':'median 15 synchronized score_candidates calls after 3 warmups; includes candidate weight extraction and prompt upload; HTTP and loading excluded'},indent=2))
    finally:runtime.manage('unload',model='a',release_host=True)


if __name__=='__main__':main()
