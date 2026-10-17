"""Real A2 HTTP admission, reclamation, cancellation and model-switching proof."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from tokenizers import Tokenizer

from damendaemon.runtime import require_gpu
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import BoundedHTTPServer, handler_for


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--chunk-size',type=int,default=0)
    parser.add_argument('--prefix-tokens',type=int,default=0)
    args=parser.parse_args()
    require_gpu()
    if os.environ.get('DAMENDAEMON_KV_CACHE_TOKENS')!='512':
        raise ValueError('this proof requires a 512-token shared pool')
    runtime=BatchedRuntime({'a':args.path,'b':args.path},context=1024,batch_size=4,
                           gpu_models=1,host_models=2,host_gib=8,queue_timeout=10,
                           prefill_chunk_size=args.chunk_size,prefix_cache_tokens=args.prefix_tokens)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        runtime.manage('load',model='a')
        tokenizer=Tokenizer.from_file(args.path+'/tokenizer.json')
        seed=tokenizer.encode('The library keeps a record of every book and its location. ').ids
        prompt=(seed*100)[:300]
        base=f'http://127.0.0.1:{server.server_port}'
        def post(alias='a',ids=prompt,count=32):
            with urlopen(Request(base+'/v1/completions',data=json.dumps({
                    'model':alias,'prompt':ids,'max_tokens':count,'temperature':0}).encode(),
                    headers={'Content-Type':'application/json'}),timeout=60) as response:
                return json.load(response)
        barrier=threading.Barrier(4)
        def concurrent(_):
            barrier.wait()
            return post()
        with ThreadPoolExecutor(4) as pool:
            outputs=list(pool.map(concurrent,range(4)))
        assert len({x['choices'][0]['text'] for x in outputs})==1
        assert all(x['damendaemon_timing']['max_batch_size']==1 for x in outputs)
        runtime.manage('status')
        executor=runtime.runtime.gpu['a']
        assert executor.pool.used_tokens==0
        peak=executor.pool.high_water_tokens
        assert 332<=peak<=512
        try:
            post(ids=(seed*100)[:500],count=64)
        except HTTPError as error:
            with error:
                if error.code != 400:
                    raise ValueError('unexpected HTTP rejection status')
        else:
            raise AssertionError('request larger than pool was accepted')
        assert post()['choices']==outputs[0]['choices']
        stream=runtime.generate('a',None,100,raw_prompt=prompt)
        next(stream)
        request=next(r for r in runtime.requests if r.command is None)
        stream.close()
        runtime.manage('status')
        assert request.cancelled.is_set() and len(request.output)<100
        assert executor.pool.used_tokens==0
        cancelled_tokens=len(request.output)
        other=post('b')
        restored=post('a')
        assert other['choices']==restored['choices']==outputs[0]['choices']
        assert restored['damendaemon_timing']['source_tier']=='host'
        runtime.manage('status')
        assert runtime.runtime.gpu['a'].pool.used_tokens==0
        print(json.dumps({'passed':True,'budget_tokens':512,'context_per_slot':1024,'slot_capacity':4,
            'prefill_chunk_size':args.chunk_size,
            'prefix_cache_tokens':args.prefix_tokens,
            'peak_reserved_tokens':peak,'capacity_pressure_requests':4,'max_active_under_pressure':1,
            'oversized_request_rejected':True,'cancelled_after_tokens':cancelled_tokens,
            'pages_after_completion_and_cancel':0,'model_switch_and_host_restore':True,
            'kernel_backend':restored['damendaemon_timing']['kernel_backend']},indent=2))
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
        runtime.close()


if __name__=='__main__':main()
