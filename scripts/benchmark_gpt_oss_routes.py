"""Matched HTTP A/B for stable expert grouping, with unchanged matrix arithmetic."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request,urlopen

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    runtime=BatchedRuntime({'oss':args.model},context=4096,batch_size=2,host_gib=18)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    rows=[]
    try:
        with runtime.runtime.session('oss') as (executor,_,_):
            base=executor.model.raw_tokens('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ')
            corpus=base*((4000+len(base)-1)//len(base))
        def select(enabled):executor.sorted_prefill_routes=enabled
        def post(ids):
            data={'model':'oss','prompt':ids,'max_tokens':16,'temperature':0}
            start=time.perf_counter()
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=json.dumps(data).encode(),
                    headers={'Content-Type':'application/json'}),timeout=240) as response:result=json.load(response)
            return result,time.perf_counter()-start
        for length,concurrency in ((16,1),(129,1),(1025,1),(4000,1),(129,2)):
            ids=corpus[:length];expected=None;measurements=[]
            def run():
                start=time.perf_counter()
                if concurrency==1:responses=[post(ids)]
                else:
                    with ThreadPoolExecutor(concurrency) as pool:responses=list(pool.map(lambda _:post(ids),range(concurrency)))
                return responses,time.perf_counter()-start
            for enabled in (False,True):
                select(enabled);responses,_=run()
                for response,_ in responses:
                    if expected is None:expected=(response['choices'],response['usage'])
                    assert (response['choices'],response['usage'])==expected,(length,concurrency,enabled)
            for trial in range(4):
                for enabled in ((False,True) if trial%2==0 else (True,False)):
                    select(enabled);torch.cuda.reset_peak_memory_stats();responses,elapsed=run()
                    for response,_ in responses:
                        assert (response['choices'],response['usage'])==expected,(length,concurrency,enabled)
                    measurements.append({'optimized':enabled,'wall_seconds':elapsed,
                        'output_tokens':sum(r['usage']['completion_tokens'] for r,_ in responses),
                        'responses':[{'http_seconds':seconds,'timing':r['dllm_timing'],'usage':r['usage']} for r,seconds in responses],
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            if concurrency>1:assert max(r['timing']['max_batch_size'] for m in measurements for r in m['responses'])==2
            row={'prompt_tokens':length,'concurrency':concurrency,'exact_responses':True,'measurements':measurements,
                'median_wall_seconds':{str(mode):statistics.median(m['wall_seconds'] for m in measurements if m['optimized']==mode)
                    for mode in (False,True)}}
            rows.append(row);print(json.dumps({'progress':row['median_wall_seconds'],'prompt_tokens':length,'concurrency':concurrency}),flush=True)
        report={'passed':True,'gpu':torch.cuda.get_device_name(),'model':args.model,'context':4096,'batch_size':2,
            'rows':rows,'optimization':'stable_expert_grouping','attention_projection_policy':'Same policy in both modes: IEEE FP32 at 256+ prefill rows, compensated below','method':'Same resident packed checkpoint, FP32 residual/KV/output and identical size-dependent attention projection policy, grouped prefill and unchanged decode graphs. Only expert route grouping changes; selected rows and their order are checked bit-exact separately. Four alternating warmed complete HTTP rounds; no simultaneous GPU workloads.'}
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'output':args.output}),flush=True)
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
