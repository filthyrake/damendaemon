"""Matched FP32 HTTP prefill A/B and bounded 8K serving on the A2."""
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
from torch.nn import functional as F
import damendaemon.qwen2 as dense
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import BoundedHTTPServer,handler_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model')
    parser.add_argument('--optimization',choices=('attention','projection','tiles'),default='attention')
    parser.add_argument('--context',type=int,default=8192);parser.add_argument('--lengths',type=int,nargs='+')
    args=parser.parse_args();lengths=args.lengths or [512,2048,args.context-32]
    if any(n<1 or n+32>args.context for n in lengths):parser.error('prompt and output must fit context')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ.update(DAMENDAEMON_DEVICE='cuda',DAMENDAEMON_LLAMA_PRECISION='float32',DAMENDAEMON_QWEN3_PRECISION='float32',
        DAMENDAEMON_SPECULATIVE_TOKENS='0',DAMENDAEMON_PREFILL_GRAPHS='0')
    torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model},context=args.context,batch_size=4)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    optimized=dense.scaled_attention;rows=[]
    try:
        with runtime.runtime.session('a') as (executor,_,_):
            assert executor.compute_dtype==torch.float32
            optimized_linear=executor.linear
            def original_linear(x,weight,bias=None):
                if args.optimization=='tiles':
                    if weight.dtype==torch.float16 and x.shape[0]>=512:
                        from damendaemon.precise_matmul import linear
                        result=linear(x.float(),weight,wide_tiles=False)
                        return result if bias is None else result+bias.float()
                    return optimized_linear(x,weight,bias)
                if weight.dtype==torch.float16 and x.shape[0]>128:
                    result=executor.kernels.linear_fp32(x.float(),weight)
                    return result if bias is None else result+bias.float()
                return optimized_linear(x,weight,bias)
            corpus=executor.model.raw_tokens(''.join(f'Record {i}: The parcel is in cabinet {i%11}, drawer {i%7}.\n' for i in range(1200)))
        def post(tokens):
            data=json.dumps({'model':'a','prompt':tokens,'max_tokens':32,'temperature':0}).encode()
            start=time.perf_counter()
            with urlopen(Request(f'http://127.0.0.1:{server.server_port}/v1/completions',data=data,
                    headers={'Content-Type':'application/json'}),timeout=180) as response:result=json.load(response)
            return result,1000*(time.perf_counter()-start)
        def select(mode):
            if args.optimization=='attention':
                dense.scaled_attention=optimized if mode else F.scaled_dot_product_attention
            else:executor.linear=optimized_linear if mode else original_linear
        for length in lengths:
            tokens=corpus[:length];assert len(tokens)==length
            native=[];baseline=[];first=[];peaks=[];expected=None
            modes=[False,True] if length<=2048 or args.optimization!='attention' else [True]
            for mode in modes:
                select(mode)
                result,_=post(tokens)
                if expected is None:expected=(result['choices'],result['usage'])
                assert (result['choices'],result['usage'])==expected
            for trial in range(4):
                for mode in (modes if trial%2==0 else list(reversed(modes))):
                    select(mode)
                    torch.cuda.reset_peak_memory_stats();result,elapsed=post(tokens)
                    assert (result['choices'],result['usage'])==expected
                    (native if mode else baseline).append(elapsed)
                    first.append({'optimized':mode,'seconds':result['damendaemon_timing']['first_token_seconds']})
                    peaks.append({'optimized':mode,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            rows.append({'prompt_tokens':length,'completion_tokens':result['usage']['completion_tokens'],
                'ordinary_ms':baseline,'optimized_ms':native,'median_ordinary_ms':statistics.median(baseline) if baseline else None,
                'median_optimized_ms':statistics.median(native),'first_token':first,'peak_allocations':peaks,
                'exact_response':True,'example':result})
        print(json.dumps({'passed':True,'model':args.model,'gpu':'NVIDIA A2','context':args.context,'optimization':args.optimization,'rows':rows,
            'method':'Same resident FP32 policy and FP16 stored matrices; four alternating full HTTP rounds after warmup. '+{
                'projection':'Compensated arithmetic versus FP32 weight expansion.',
                'tiles':'Wider output tiles versus the original compensated-kernel launch shape; arithmetic is otherwise identical.',
                'attention':'Fused versus math attention; above2048 prompt tokens only fused is measured after the original full8K allocation failure.'}[args.optimization]},indent=2))
    finally:
        dense.scaled_attention=optimized
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
