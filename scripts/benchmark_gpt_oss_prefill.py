"""Matched raw HTTP A/B of routed versus grouped GPT-OSS prompt processing."""
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--lengths',type=int,nargs='+',default=[16,32,129,224])
    parser.add_argument('--rounds',type=int,default=4);parser.add_argument('--output',required=True);args=parser.parse_args()
    if any(n<1 or n+8>256 for n in args.lengths) or args.rounds<1:parser.error('invalid workload')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ.update(DLLM_DEVICE='cuda',DLLM_GPT_OSS_PREFILL='grouped');torch.set_num_threads(4)
    runtime=BatchedRuntime({'oss':args.model},context=256,batch_size=2,host_gib=18)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(ids):
        body={'model':'oss','prompt':ids,'max_tokens':8,'temperature':0}
        start=time.perf_counter()
        with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}),timeout=240) as response:result=json.load(response)
        return result,time.perf_counter()-start
    rows=[]
    try:
        with runtime.runtime.session('oss') as (executor,_,_):
            base=executor.model.raw_tokens('The capital of France is Paris. Write a short explanation of how the ocean affects coastal weather. ')
            corpus=base*((max(args.lengths)+len(base)-1)//len(base))
        for length in args.lengths:
            ids=corpus[:length];expected=None;measurements=[]
            for mode in ('routed','grouped'):
                executor.grouped_prefill=mode;result,_=post(ids)
                if expected is None:expected=(result['choices'],result['usage'])
                assert (result['choices'],result['usage'])==expected,(length,mode)
            for trial in range(args.rounds):
                for mode in (('routed','grouped') if trial%2==0 else ('grouped','routed')):
                    executor.grouped_prefill=mode;torch.cuda.reset_peak_memory_stats()
                    result,elapsed=post(ids)
                    assert (result['choices'],result['usage'])==expected,(length,mode)
                    measurements.append({'mode':mode,'http_seconds':elapsed,
                        'first_token_seconds':result['dllm_timing']['first_token_seconds'],
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            rows.append({'prompt_tokens':length,'completion_tokens':result['usage']['completion_tokens'],
                'exact_response':True,'measurements':measurements,'median_seconds':{
                    mode:statistics.median(x['http_seconds'] for x in measurements if x['mode']==mode)
                    for mode in ('routed','grouped')}})
            print(json.dumps({'progress':rows[-1]}),flush=True)
        report={'passed':True,'gpu':torch.cuda.get_device_name(),'model':args.model,'context':256,
            'batch_size':2,'rounds':args.rounds,'rows':rows,
            'method':'Same resident model, FP32 residual/KV/output policy, packed checkpoint and CUDA decode graphs. Grouped experts use lossless FP16 expansion with compensated Tensor Core products; routed experts reduce FP32 products. Alternating warmed full HTTP rounds with identical generated responses.'}
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'output':args.output}),flush=True)
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
