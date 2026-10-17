"""A2 prefill graph correctness, bounded lifetime and complete prefill timings."""
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
from damendaemon.runtime import Runtime


def timed(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(15):
        start=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return statistics.median(times)


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    os.environ['DAMENDAEMON_PREFILL_GRAPHS']='2';os.environ['DAMENDAEMON_PREFILL_GRAPH_MAX_TOKENS']='128'
    torch.set_num_threads(4);runtime=Runtime({'a':args.model},context=256,batch_size=4)
    checks=[]
    try:
        with runtime.session('a') as (executor,_,_):
            cache=executor.prefill_graph_cache;assert cache is not None
            for count,slot,value in [(16,0,40),(16,0,100),(16,1,40),(32,0,100),(16,0,200),(128,2,40),(129,3,100)]:
                prompt=[value]*count
                executor.prefill_graph_cache=None
                expected=executor.prefill(prompt,slot).clone()
                next_token=expected.argmax().item()
                expected_decode=executor.decode_batch([next_token],[slot])[0].clone()
                executor.prefill_graph_cache=cache
                before=cache.captures;start=time.perf_counter()
                actual=executor.prefill(prompt,slot).clone();torch.cuda.synchronize()
                first_ms=(time.perf_counter()-start)*1000
                torch.testing.assert_close(actual,expected,atol=.0001,rtol=.0001)
                actual_decode=executor.decode_batch([next_token],[slot])[0]
                torch.testing.assert_close(actual_decode,expected_decode,atol=.0001,rtol=.0001)
                assert len(cache.entries)<=2
                if count>128:assert before==cache.captures
                checks.append({'count':count,'slot':slot,'value':value,'first_call_ms':first_ms,
                    'max_logit_error':(actual-expected).abs().max().item(),
                    'max_decode_error':(actual_decode-expected_decode).abs().max().item()})
            # A narrow scoring request must use a distinct candidate-keyed graph.
            before=cache.hits
            actual=executor.score_candidates([40]*16,[0,1,100])
            assert actual.shape[-1]==3 and cache.hits==before
            repeated=executor.score_candidates([40]*16,[0,1,100])
            assert repeated.shape[-1]==3 and cache.hits==before+1
            rows=[]
            for count in (16,128):
                prompt=[40]*count
                executor.prefill_graph_cache=None;eager=timed(lambda:executor.prefill(prompt))
                executor.prefill_graph_cache=cache;replay=timed(lambda:executor.prefill(prompt))
                rows.append({'tokens':count,'eager_ms':eager,'graph_ms':replay,'speedup':eager/replay})
            # Reference only weakly: graph ownership must end when the entry is evicted.
            oldest=next(iter(cache.entries));ref=weakref.ref(cache.entries[oldest][2])
            executor.prefill([40]*48,slot=3);gc.collect();assert ref() is None
            assert cache.evictions>0 and cache.hits>0
            torch.cuda.synchronize();reserved_with=torch.cuda.memory_reserved();allocated_with=torch.cuda.memory_allocated()
            refs=[weakref.ref(entry[2]) for entry in cache.entries.values()]
            cache.entries.clear();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            assert all(ref() is None for ref in refs)
            memory={'allocated_with_cache':allocated_with,'allocated_after_clear':torch.cuda.memory_allocated(),
                'reserved_with_cache':reserved_with,'reserved_after_clear':torch.cuda.memory_reserved()}
            assert memory['allocated_after_clear']<allocated_with
            stats={'hits':cache.hits,'captures':cache.captures,'evictions':cache.evictions}
            # Recreate a cached entry and release all test-owned executor references.
            executor.prefill([40]*16);executor_ref=weakref.ref(executor)
            graph_ref=weakref.ref(next(iter(cache.entries.values()))[2])
            del executor,cache
        runtime.manage('unload',model='a');gc.collect()
        assert executor_ref() is None and graph_ref() is None
        print(json.dumps({'passed':True,'model':args.model,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'checks':checks,'timings':rows,'cache':stats,'memory':memory,
            'eviction_frees_graph':True,'unload_frees_executor_and_graph':True,'scoring_uses_distinct_graph':True,
            'timing':'median 15 synchronized complete prefills after 3 warmups, includes prompt tensor creation/copy; loading and HTTP excluded'},indent=2))
    finally:runtime.manage('unload',model='a',release_host=True)


if __name__=='__main__':main()
