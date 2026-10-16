"""A2 HTTP precision tradeoff; modes may produce different greedy text."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import statistics
import threading
import time
from urllib.request import Request,urlopen

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


PROMPTS={
    'repetition':'Write the following sentence exactly ten times, with no introduction: The quick brown fox jumps over the lazy dog.',
    'explanation':'Explain why the sky looks blue in simple language.',
    'code':'Write a Python function that returns a list containing the integers from 1 to 30, written out explicitly.',
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    torch.set_num_threads(4);os.environ['DLLM_DEVICE']='cuda';reports=[]
    for precision,width in (('mixed',0),('float32',0),('float32',8)):
        os.environ['DLLM_LLAMA_PRECISION']=precision;os.environ['DLLM_SPECULATIVE_TOKENS']=str(width)
        runtime=BatchedRuntime({'a':args.model},context=512,batch_size=4)
        server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            rows=[]
            for name,prompt in PROMPTS.items():
                body=json.dumps({'model':'a','messages':[{'role':'user','content':prompt}],
                    'temperature':0,'max_tokens':64}).encode()
                times=[];counts=[];outputs=[]
                for trial in range(7):
                    start=time.perf_counter()
                    with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                            data=body,headers={'Content-Type':'application/json'}),timeout=180) as response:
                        result=json.load(response)
                    elapsed=1000*(time.perf_counter()-start)
                    if trial:
                        times.append(elapsed);counts.append(result['usage']['completion_tokens'])
                        outputs.append(result['choices'][0]['message']['content'])
                assert len(set(outputs))==1 and len(set(counts))==1
                rows.append({'case':name,'prompt':prompt,'milliseconds':times,'median_ms':statistics.median(times),
                    'completion_tokens':counts[0],'output':outputs[0],'timing':result['dllm_timing']})
            reports.append({'precision':precision,'speculative_width':width,'rows':rows})
        finally:
            server.shutdown();server.server_close();thread.join();runtime.close()
    print(json.dumps({'model':args.model,'gpu':'NVIDIA A2','context':512,'reports':reports,
        'method':'Sequential modes, one warmup and six measured HTTP requests per case. Precision changes can change outputs; this is a policy tradeoff, not a matched-precision speedup.'},indent=2))


if __name__=='__main__':main()
