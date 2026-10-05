"""Real native K-quant executor/HTTP proof against the same expanded coefficients."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from urllib.request import Request,urlopen

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.quants import dequantize
from dllm.bench import events
from dllm.device import device_name, empty_cache
from dllm.gguf_model import GgufWeights
from dllm.runtime import require_gpu, make_executor
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer, handler_for


def expanded(model):
    """Use independent format decoding; do not share native GPU unpack arithmetic."""
    assert not model.scales, 'this proof expects all quantized matrices to use raw layouts'
    reference=copy.copy(model); weights={}; copies={}
    for name,value in model.weights.items():
        if id(value) not in copies:
            if name in model.packed_formats:
                width,spec=model.packed_formats[name]; raw=value.numpy(); chunks=[]
                for first,last,kind,offset in spec:
                    block,size=GGML_QUANT_SIZES[GGMLQuantizationType(kind)]
                    count=(last-first)*width//block*size
                    decoded=dequantize(raw[offset:offset+count].reshape(-1,size),GGMLQuantizationType(kind))
                    chunks.append(torch.from_numpy(decoded.copy()).reshape(last-first,width).half())
                copies[id(value)]=torch.cat(chunks)
            else:copies[id(value)]=value
        weights[name]=copies[id(value)]
    reference.packed_formats={};reference.quantization='none'
    reference.weight_format='expanded_same_kquant_f16_reference'
    reference.finalize_weights(weights,{})
    return reference


def main():
    p=argparse.ArgumentParser();p.add_argument('model')
    p.add_argument('--experimental-residual',action='store_true')
    p.add_argument('--executor-only',action='store_true',help='skip HTTP checks and timings');args=p.parse_args()
    experiment=None
    if args.experimental_residual:
        from experiments.kquant_residual import install_validation_dispatch
        experiment=install_validation_dispatch()
    require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=GgufWeights(args.model,512,'none');reference=expanded(model)
    report={'model':str(Path(args.model).resolve()),'device':device_name(),'torch':torch.__version__,
            'packed_host_bytes':model.bytes,'expanded_host_bytes':reference.bytes,'comparisons':[],
            'experiment':experiment,'native_library':os.environ.get('DLLM_KQUANT_CUDA_LIBRARY')}
    def compare(expected,actual,**case):
        error=actual.float()-expected.float()
        absolute=error.abs().max().item()
        relative=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item()
        equal=torch.equal(expected.argmax(-1),actual.argmax(-1))
        report['comparisons'].append({**case,'max_absolute':absolute,'relative_rms':relative,'greedy_equal':equal})
        assert absolute<.5 and relative<.01 and equal,report['comparisons'][-1]
    with torch.inference_mode():
        for paged in (False,True):
            if paged:os.environ['DLLM_KV_CACHE_TOKENS']='1024'
            else:os.environ.pop('DLLM_KV_CACHE_TOKENS',None)
            for graph in (False,True):
                print(f'Checking batching/chunking: paged={paged}, graph={graph}',file=sys.stderr,flush=True)
                a=make_executor(reference,use_graph=graph,max_batch=4)
                b=make_executor(model,use_graph=graph,max_batch=4)
                report['packed_backend']=b.kernel_backend
                report['packed_workspace_reserve_bytes']=b.packed_workspace_bytes
                prompts=[[9707,11,1879,0]*size for size in (1,4,8,16)]
                for count in (1,4):
                    slots=list(range(count))
                    expected=a.prefill_batch(prompts[:count],slots)
                    actual=b.prefill_batch(prompts[:count],slots)
                    for step in range(8):
                        compare(expected,actual,paged=paged,graph=graph,kind=f'batch{count}',step=step)
                        if step<7:
                            tokens=expected.argmax(-1).tolist()
                            expected,_=a.decode_batch(tokens,slots);actual,_=b.decode_batch(tokens,slots)
                for start in range(0,128,32):
                    expected=a.prefill_chunk([9707,11,1879,0]*8,0,start,final=start==96)
                    actual=b.prefill_chunk([9707,11,1879,0]*8,0,start,final=start==96)
                compare(expected,actual,paged=paged,graph=graph,kind='chunked')
                del a,b,expected,actual
                gc.collect();empty_cache()
    os.environ.pop('DLLM_KV_CACHE_TOKENS',None)
    if args.executor_only:
        report.update(passed=True,scope='executor only; no HTTP checks')
        print(json.dumps(report,indent=2))
        return
    baseline_tokens=model.raw_tokens('Binary search repeatedly halves a sorted range. A detailed explanation follows: ')
    del model
    gc.collect();empty_cache()
    runtime=BatchedRuntime({},context=512,batch_size=4,gpu_models=2,host_models=2,host_gib=12)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(route,body):
        return urlopen(Request(base+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300)
    def request(route,body):
        with post(route,body) as response:return json.load(response)
    def chat(alias='packed',**options):
        return dict(model=alias,messages=[{'role':'user','content':'Explain binary search with a short worked example.'}],
                    temperature=0,max_tokens=32,**options)
    try:
        print('Checking HTTP chat, streaming, concurrency and lifecycle',file=sys.stderr,flush=True)
        for alias in ('packed','expanded'):
            request('/v1/models/register',dict(model=alias,path=str(Path(args.model).resolve()),quantization='none'))
        # The comparator exists only in this validation process. Production
        # registration and every packed reload still use the normal file loader.
        with runtime.runtime.lock:runtime.runtime.host['expanded']=reference
        whole=request('/v1/chat/completions',chat())
        other=request('/v1/chat/completions',chat('expanded'))
        expected=whole['choices'][0]['message']['content']
        assert other['choices'][0]['message']['content']==expected,(whole,other)
        with post('/v1/chat/completions',chat(stream=True)) as response:chunks=list(events(response))
        assert chunks[-1]=='[DONE]'
        text=''.join(c.get('delta',{}).get('content','') for x in chunks[:-1] for c in json.loads(x)['choices'])
        assert text==expected
        with ThreadPoolExecutor(4) as pool:
            concurrent=list(pool.map(lambda _:request('/v1/chat/completions',chat()),range(4)))
        assert all(r['choices'][0]['message']['content']==expected for r in concurrent)
        with post('/v1/chat/completions',chat(stream=True)) as response:next(events(response))
        assert request('/v1/chat/completions',chat())['choices'][0]['message']['content']==expected
        request('/v1/models/unload',dict(model='packed'))
        restored=request('/v1/chat/completions',chat())
        assert restored['dllm_timing']['source_tier']=='host'
        assert restored['choices'][0]['message']['content']==expected
        trials=[]
        for length,concurrency in ((16,1),(129,1),(400,1),(129,2)):
            print(f'Timing HTTP: input={length}, concurrency={concurrency}',file=sys.stderr,flush=True)
            ids=(baseline_tokens*((length+len(baseline_tokens)-1)//len(baseline_tokens)))[:length]
            def run(alias):
                body=dict(model=alias,prompt=ids,max_tokens=16,temperature=0)
                start=time.perf_counter()
                if concurrency==1:values=[request('/v1/completions',body)]
                else:
                    with ThreadPoolExecutor(concurrency) as pool:
                        values=list(pool.map(lambda _:request('/v1/completions',body),range(concurrency)))
                seconds=time.perf_counter()-start
                assert all(r['usage']['prompt_tokens']==length and r['usage']['completion_tokens']==16 for r in values),values
                return dict(alias=alias,prompt_tokens=length,concurrency=concurrency,seconds=seconds,
                            texts=[r['choices'][0]['text'] for r in values],usage=[r['usage'] for r in values])
            for alias in ('packed','expanded'):
                for _ in range(3):run(alias)
            for turn in range(6):
                pair=[run(alias) for alias in (('packed','expanded') if turn%2==0 else ('expanded','packed'))]
                assert pair[0]['texts']==pair[1]['texts'],pair
                trials.extend(pair)
        request('/v1/models/unload',dict(model='packed',release_host=True))
        reloaded=request('/v1/chat/completions',chat())
        assert reloaded['dllm_timing']['source_tier']=='disk'
        assert reloaded['choices'][0]['message']['content']==expected
        report['http']={'registration':True,'chat_matches_expanded':True,'stream':True,'concurrent_requests':4,
            'cancellation_recovery':True,'host_restore':True,'disk_reload':True,'sample':expected,'trials':trials,
            'medians':[dict(prompt_tokens=length,concurrency=count,**{alias:statistics.median(
                t['seconds'] for t in trials if t['alias']==alias and t['prompt_tokens']==length and t['concurrency']==count)
                for alias in ('packed','expanded')}) for length,count in ((16,1),(129,1),(400,1),(129,2))]}
        report['passed']=True
        print(json.dumps(report,indent=2))
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()


if __name__=='__main__':main()
