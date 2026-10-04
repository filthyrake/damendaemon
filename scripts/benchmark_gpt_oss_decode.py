"""Matched graph-to-graph HTTP A/B for chunked MXFP4 expert reductions."""
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
    runtime=BatchedRuntime({'oss':args.model},context=256,batch_size=2,host_gib=18)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    rows=[]
    try:
        with runtime.runtime.session('oss') as (executor,_,_):
            # Both graph sets share one resident model and KV allocation. Save
            # each graph's own stable metadata/workspace when recapturing.
            graph_sets={True:dict(executor.batch_graphs)}
            executor.chunked_experts=False;executor.capture_graphs()
            graph_sets[False]=dict(executor.batch_graphs)
            base=executor.model.raw_tokens('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ')
            corpus=base*((129+len(base)-1)//len(base))
        def select(chunked):
            executor.chunked_experts=chunked
            executor.batch_graphs=graph_sets[chunked]
        def post(ids):
            data={'model':'oss','prompt':ids,'max_tokens':32,'temperature':0}
            start=time.perf_counter()
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=json.dumps(data).encode(),
                    headers={'Content-Type':'application/json'}),timeout=240) as response:result=json.load(response)
            return result,time.perf_counter()-start
        for length,concurrency in ((5,1),(32,1),(129,1),(32,2)):
            ids=corpus[:length];expected=None;measurements=[]
            def run():
                start=time.perf_counter()
                if concurrency==1:responses=[post(ids)]
                else:
                    with ThreadPoolExecutor(concurrency) as pool:responses=list(pool.map(lambda _:post(ids),range(concurrency)))
                return responses,time.perf_counter()-start
            for chunked in (False,True):
                select(chunked);responses,_=run()
                for response,_ in responses:
                    if expected is None:expected=(response['choices'],response['usage'])
                    assert (response['choices'],response['usage'])==expected,(length,concurrency,chunked)
            for trial in range(4):
                for chunked in ((False,True) if trial%2==0 else (True,False)):
                    select(chunked);torch.cuda.reset_peak_memory_stats();responses,elapsed=run()
                    for response,_ in responses:
                        assert (response['choices'],response['usage'])==expected,(length,concurrency,chunked)
                    measurements.append({'chunked':chunked,'wall_seconds':elapsed,
                        'output_tokens':sum(r['usage']['completion_tokens'] for r,_ in responses),
                        'responses':[{'http_seconds':seconds,'timing':r['dllm_timing'],'usage':r['usage']} for r,seconds in responses],
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            if concurrency>1:assert max(r['timing']['max_batch_size'] for m in measurements for r in m['responses'])==2
            row={'prompt_tokens':length,'concurrency':concurrency,'exact_responses':True,'measurements':measurements,
                'median_wall_seconds':{str(mode):statistics.median(m['wall_seconds'] for m in measurements if m['chunked']==mode)
                    for mode in (False,True)}}
            rows.append(row);print(json.dumps({'progress':row['median_wall_seconds'],'prompt_tokens':length,'concurrency':concurrency}),flush=True)
        report={'passed':True,'gpu':torch.cuda.get_device_name(),'model':args.model,'context':256,'batch_size':2,
            'rows':rows,'method':'Same resident packed checkpoint, FP32 residual/KV/output policy and grouped prefill. Separate baseline/full and optimized/chunked CUDA graph sets share weights/KV. Four alternating warmed complete HTTP rounds; no simultaneous GPU workloads.'}
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'output':args.output}),flush=True)
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
