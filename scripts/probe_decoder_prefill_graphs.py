"""Isolated CUDA prefill graph probe; does not enable a production fast path."""
import argparse
import gc
import json
import statistics
import time
import torch
from dllm.runtime import Runtime


def measure(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(25):
        start=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return statistics.median(times)


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    torch.set_num_threads(4);runtime=Runtime({'probe':args.model},context=1024,use_graph=True)
    reports=[]
    try:
        with runtime.session('probe') as (executor,_,_):
            for count in (16,128,512):
                ids=torch.tensor(([40,298,1111,13]*((count+3)//4))[:count],device='cuda')
                positions=torch.arange(count,device='cuda')
                def forward():return executor.forward(ids,positions,prefill=True)
                expected=forward().clone()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):forward()
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                before=torch.cuda.memory_allocated();start=time.perf_counter()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):result=forward()
                torch.cuda.synchronize();capture=(time.perf_counter()-start)*1000
                retained=torch.cuda.memory_allocated()-before
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(result,expected,atol=.0001,rtol=.0001)
                # Replay must consume changed IDs, not capture a constant prompt.
                ids.fill_(100)
                changed=forward().clone();graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(result,changed,atol=.0001,rtol=.0001)
                eager=measure(forward);replay=measure(graph.replay)
                reports.append({'tokens':count,'eager_ms':eager,'graph_ms':replay,
                    'speedup':eager/replay,'capture_ms':capture,'retained_allocated_bytes':retained,
                    'changed_prompt_equal':True,'max_logit_error':(result-changed).abs().max().item()})
                del graph,result,expected,changed;gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'model':args.model,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
                'compute_dtype':str(executor.compute_dtype),'rows':reports,
                'timing':'median 25 synchronized complete forward/replay calls after 5 warmups; device IDs already resident; HTTP and input copy excluded'},indent=2))
    finally:runtime.manage('unload',model='probe',release_host=True)


if __name__=='__main__':main()
