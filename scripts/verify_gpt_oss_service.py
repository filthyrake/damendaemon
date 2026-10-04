"""Raw GPT-OSS HTTP, streaming, batching and hot-swap proof on A2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import weakref

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('swap_model');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    runtime=BatchedRuntime({'oss':args.model,'swap':args.swap_model},context=256,batch_size=2,
        host_gib=18,gpu_models=1,host_models=2)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint=f'http://127.0.0.1:{server.server_port}'
    data={'model':'oss','prompt':'The capital of France is','max_tokens':16,'temperature':0}
    def post(body,path='/v1/completions'):
        start=time.perf_counter()
        with urlopen(Request(endpoint+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=240) as response:
            result=json.load(response)
        return result,time.perf_counter()-start
    def equal(actual,expected):
        assert actual['choices']==expected['choices']
        assert actual['usage']==expected['usage']
    def stream(disconnect=False):
        content='';done=False
        with urlopen(Request(endpoint+'/v1/completions',data=json.dumps({**data,'stream':True}).encode(),
                headers={'Content-Type':'application/json'}),timeout=240) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                line=line[6:].strip()
                if line==b'[DONE]':done=True;break
                event=json.loads(line);assert 'error' not in event,event
                content+=''.join(choice.get('text','') for choice in event.get('choices',[]))
                if disconnect and content:break
        if not disconnect:assert done
        return content
    try:
        reference,cold=post(data);warm,warm_seconds=post(data);equal(warm,reference)
        assert warm['dllm_timing']['weight_quantization']=='mxfp4'
        assert stream()==reference['choices'][0]['text']
        stream(disconnect=True);equal(post(data)[0],reference)
        with ThreadPoolExecutor(2) as pool:concurrent=list(pool.map(lambda _:post(data)[0],range(2)))
        for result in concurrent:equal(result,reference)
        assert max(r['dllm_timing']['max_batch_size'] for r in concurrent)==2
        try:
            post({'model':'oss','messages':[{'role':'user','content':'Hello'}],'max_tokens':4},'/v1/chat/completions')
        except HTTPError as error:
            message=error.read().decode();assert error.code==400 and 'error' in json.loads(message),message
        else:raise AssertionError('unqualified Harmony route accepted')
        with runtime.runtime.session('oss') as (executor,_,_):
            old=weakref.ref(executor);graph=weakref.ref(executor.batch_graphs[1][4])
        del executor
        swap,_=post({**data,'model':'swap','max_tokens':2})
        gc.collect();torch.cuda.synchronize();assert old() is None and graph() is None
        restored,restore_seconds=post(data);equal(restored,reference)
        assert restored['dllm_timing']['source_tier']=='host'
        runtime.manage('unload',model='oss',release_host=False)
        again,_=post(data);equal(again,reference);assert again['dllm_timing']['source_tier']=='host'
        print(json.dumps({'passed':True,'gpu':torch.cuda.get_device_name(),'models':{'oss':args.model,'swap':args.swap_model},
            'context':256,'batch_size':2,'reference':reference,'warm':warm,'cold_http_seconds':cold,
            'warm_http_seconds':warm_seconds,'host_restore_http_seconds':restore_seconds,
            'stream_equivalence':True,'disconnect_recovery':True,'concurrent_equivalence':True,
            'unqualified_harmony_rejected':True,'graph_executor_reclaimed_on_swap':True,
            'host_swap_restore_equivalence':True,'explicit_unload_restore':True,'native_format_reported':True},indent=2))
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
