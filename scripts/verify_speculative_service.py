"""Real A2 HTTP A/B, fallback, streaming and residency proof for speculation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import weakref

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    p=argparse.ArgumentParser();p.add_argument('first_model');p.add_argument('second_model',nargs='?')
    p.add_argument('--first-name',default='qwen2');p.add_argument('--second-name',default='qwen3')
    p.add_argument('--only',nargs='+');args=p.parse_args()
    models={args.first_name:args.first_model}
    if args.second_model:
        if args.second_name in models:p.error('model names must be distinct')
        models[args.second_name]=args.second_model
    selected=args.only or list(models)
    if any(name not in models for name in selected):p.error('unknown selected model name')
    families={name:json.loads((Path(path)/'config.json').read_text())['model_type'] for name,path in models.items()}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']='cuda';os.environ['DLLM_SPECULATIVE_TOKENS']='8';torch.set_num_threads(4)
    runtime=BatchedRuntime(models,context=512,batch_size=4,gpu_models=1,host_models=2,host_gib=8)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    repetition='Write the following sentence exactly ten times, with no introduction: The quick brown fox jumps over the lazy dog.'
    reports=[]
    def body(alias,text=repetition,**changes):
        result={'model':alias,'messages':[{'role':'user','content':text}],'max_tokens':64,'temperature':0}
        if families[alias]=='qwen3':result['chat_template_kwargs']={'enable_thinking':False}
        return {**result,**changes}
    def post(data):
        start=time.perf_counter()
        try:
            with urlopen(Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:result=json.load(r)
        except HTTPError as error:raise RuntimeError(error.read().decode()) from error
        return result,(time.perf_counter()-start)*1000
    @contextmanager
    def baseline(alias):
        with runtime.runtime.session(alias) as (executor,_,_):
            verifier=executor.speculative_verifier;assert verifier is not None
            executor.speculative_verifier=None
        try:yield
        finally:
            with runtime.runtime.session(alias) as (executor,_,_):executor.speculative_verifier=verifier
    def equal(left,right):
        assert left['choices'][0]['message']==right['choices'][0]['message']
        assert left['choices'][0]['finish_reason']==right['choices'][0]['finish_reason']
        assert left['usage']==right['usage']
    def stream(data,disconnect=False):
        content='';data={**data,'stream':True}
        with urlopen(Request(url,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=180) as r:
            for raw in r:
                if not raw.startswith(b'data: '):continue
                line=raw[6:].strip()
                if line==b'[DONE]':break
                event=json.loads(line)
                assert 'error' not in event,event
                for choice in event.get('choices',[]):content+=choice.get('delta',{}).get('content','')
                if disconnect and content:break
        return content
    try:
        for alias in selected:
            rows=[]
            with runtime.runtime.session(alias) as (executor,_,_):
                precision={'compute_dtype':str(executor.compute_dtype),
                    'residual_dtype':str(executor.residual_dtype),'kv_dtype':str(executor.kv[0].dtype),
                    'matrix_storage_dtype':str(executor.w['head'].dtype),
                    'qwen3_precision':executor.qwen3_precision,'llama_precision':executor.llama_precision,
                    'dense_precision':executor.dense_precision}
            del executor
            for name,text in [('repetition',repetition),('explanation','Explain why the sky looks blue in simple language.'),
                    ('code','Write a Python function that returns a list containing the integers from 1 to 30, written out explicitly.')]:
                print(alias+' '+name,file=__import__('sys').stderr,flush=True)
                data=body(alias,text)
                with baseline(alias):reference,_=post(data)
                candidate,_=post(data);equal(candidate,reference)
                if name=='repetition':assert candidate['dllm_timing']['speculative_matched_drafts']>0
                normal=[];speculative=[]
                for step in range(6):
                    for enabled in ([False,True] if step%2==0 else [True,False]):
                        if enabled:result,elapsed=post(data);speculative.append(elapsed)
                        else:
                            with baseline(alias):result,elapsed=post(data)
                            normal.append(elapsed)
                        equal(result,reference)
                rows.append({'case':name,'baseline_ms':normal,'speculative_ms':speculative,
                    'median_baseline_ms':statistics.median(normal),'median_speculative_ms':statistics.median(speculative),
                    'example':candidate,'exact_response':True})
            # Unsupported sampling/logprob/stop/grammar combinations must retain
            # ordinary decode behavior even while the verifier is resident.
            fallback=[]
            for changes in ({'temperature':.7,'seed':123},{'frequency_penalty':.5},
                    {'logprobs':True,'top_logprobs':0},{'stop':['brown']},
                    {'response_format':{'type':'json_schema','json_schema':{'name':'answer','schema':
                        {'type':'object','properties':{'answer':{'type':'string','const':'ok'}},
                         'required':['answer'],'additionalProperties':False}}}}):
                data=body(alias,**changes)
                with baseline(alias):reference,_=post(data)
                result,_=post(data);equal(result,reference)
                assert result['dllm_timing']['speculative_blocks']==0
                fallback.append({'options':changes,'passed':True})
            data=body(alias)
            result,_=post(data)
            assert stream(data)==result['choices'][0]['message']['content']
            stream(data,disconnect=True)
            equal(post(data)[0],result)
            # Slow consumption fills the bounded event queue. A later consumer
            # must receive the same tokens rather than overflow or skip drafts.
            messages=[{'role':'user','content':repetition}]
            kwargs={'enable_thinking':False} if families[alias]=='qwen3' else {}
            def direct():return runtime.generate(alias,messages,64,**kwargs)
            with baseline(alias):expected=[e['token'] for e in direct() if 'token' in e]
            generator=direct();first=next(generator);time.sleep(.1)
            tokens=([first['token']] if 'token' in first else [])+[e['token'] for e in generator if 'token' in e]
            assert tokens==expected
            with ThreadPoolExecutor(4) as pool:
                with baseline(alias):references=list(pool.map(lambda _:post(data)[0],range(4)))
                concurrent=list(pool.map(lambda _:post(data)[0],range(4)))
            for actual,expected_response in zip(concurrent,references):equal(actual,expected_response)
            assert max(r['dllm_timing']['max_batch_size'] for r in concurrent)>1
            with runtime.runtime.session(alias) as (executor,_,_):
                graph_ref=weakref.ref(executor.speculative_verifier.graph)
                verifier_ref=weakref.ref(executor.speculative_verifier)
            del executor
            runtime.manage('unload',model=alias,release_host=False);gc.collect();torch.cuda.synchronize()
            assert graph_ref() is None and verifier_ref() is None
            restored,_=post(data);equal(restored,result)
            assert restored['dllm_timing']['source_tier']=='host'
            reports.append({'model':alias,'precision':precision,'rows':rows,'fallback':fallback,'stream_equivalence':True,
                'disconnect_recovery':True,'backpressure':True,'concurrent_equivalence':True,
                'graph_reclaimed_on_unload':True,'host_restore':True})
        print(json.dumps({'passed':True,'gpu':'NVIDIA A2','speculative_width':8,
            'models':models,'reports':reports,
            'timing':'Same server and resident model, six alternating A/B measured rounds after warmup; complete loopback HTTP'},indent=2))
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
