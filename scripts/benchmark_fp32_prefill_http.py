"""Matched FP32 HTTP prefill A/B and bounded 8K serving on the A2."""
import argparse
import json
import os
import statistics
import threading
import time
from urllib.request import Request,urlopen

import torch
from torch.nn import functional as F
import dllm.qwen2 as dense
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ.update(DLLM_DEVICE='cuda',DLLM_LLAMA_PRECISION='float32',DLLM_QWEN3_PRECISION='float32',
        DLLM_SPECULATIVE_TOKENS='0',DLLM_PREFILL_GRAPHS='0')
    torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model},context=8192,batch_size=4)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    optimized=dense.scaled_attention;rows=[]
    try:
        with runtime.runtime.session('a') as (executor,_,_):
            assert executor.compute_dtype==torch.float32
            corpus=executor.model.raw_tokens(''.join(f'Record {i}: The parcel is in cabinet {i%11}, drawer {i%7}.\n' for i in range(1200)))
        def post(tokens):
            data=json.dumps({'model':'a','prompt':tokens,'max_tokens':32,'temperature':0}).encode()
            start=time.perf_counter()
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=data,
                    headers={'Content-Type':'application/json'}),timeout=180) as response:result=json.load(response)
            return result,1000*(time.perf_counter()-start)
        for length in (512,2048,8160):
            tokens=corpus[:length];assert len(tokens)==length
            native=[];baseline=[];first=[];peaks=[];expected=None
            modes=[False,True] if length<=2048 else [True]
            for mode in modes:
                dense.scaled_attention=optimized if mode else F.scaled_dot_product_attention
                result,_=post(tokens)
                if expected is None:expected=(result['choices'],result['usage'])
                assert (result['choices'],result['usage'])==expected
            for trial in range(4):
                for mode in (modes if trial%2==0 else list(reversed(modes))):
                    dense.scaled_attention=optimized if mode else F.scaled_dot_product_attention
                    torch.cuda.reset_peak_memory_stats();result,elapsed=post(tokens)
                    assert (result['choices'],result['usage'])==expected
                    (native if mode else baseline).append(elapsed)
                    first.append({'optimized':mode,'seconds':result['dllm_timing']['first_token_seconds']})
                    peaks.append({'optimized':mode,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            rows.append({'prompt_tokens':length,'completion_tokens':result['usage']['completion_tokens'],
                'ordinary_ms':baseline,'optimized_ms':native,'median_ordinary_ms':statistics.median(baseline) if baseline else None,
                'median_optimized_ms':statistics.median(native),'first_token':first,'peak_allocations':peaks,
                'exact_response':True,'example':result})
        print(json.dumps({'passed':True,'model':args.model,'gpu':'NVIDIA A2','rows':rows,
            'method':'Same resident FP32 executor; four alternating full HTTP rounds after warmup. At 8160 prompt tokens, optimized only: the original full 8K math attention exhausted A2 memory.'},indent=2))
    finally:
        dense.scaled_attention=optimized
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
