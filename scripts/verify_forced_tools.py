"""Real GPU proof of single schema-constrained required/named tool calls."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import copy
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
import torch
from transformers import AutoTokenizer
from damendaemon.bench import events
from damendaemon.runtime import require_gpu
from damendaemon.scheduler import BatchedRuntime
from damendaemon.server import BoundedHTTPServer,handler_for
from damendaemon.tool_calls import prepare_tools
from damendaemon.structured import matches_schema

TOOLS=[{'type':'function','function':{'name':name,'description':description,'strict':True,
    'parameters':{'type':'object','properties':{key:{'type':'string','const':value}},
        'required':[key],'additionalProperties':False}}}
    for name,description,key,value in [('weather','Get weather for a city','city','Paris'),
        ('translate','Translate a phrase','phrase','bonjour')]]
MESSAGES=[{'role':'user','content':'Please get the weather in Paris.'}]


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--other-model',required=True)
    p.add_argument('--device',choices=('cuda','mps'),required=True);args=p.parse_args()
    os.environ['DAMENDAEMON_DEVICE']=args.device;require_gpu();torch.set_num_threads(4)
    runtime=BatchedRuntime({'a':args.model,'b':args.other_model},context=2048,batch_size=4,
        gpu_models=1,host_models=2,host_gib=16)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    def post(alias='a',**changes):
        body={'model':alias,'messages':MESSAGES,'tools':TOOLS,'tool_choice':'required',
            'temperature':0,'max_tokens':96,**changes}
        return urlopen(Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180)
    def request(alias='a',**changes):
        with post(alias,**changes) as response:return json.load(response)
    def check(result,choice='required'):
        item=result['choices'][0];assert item['finish_reason']=='tool_calls',result
        assert item['message']['content'] is None,result
        calls=item['message']['tool_calls'];assert len(calls)==1,calls
        call=calls[0];value={'name':call['function']['name'],'arguments':json.loads(call['function']['arguments'])}
        assert matches_schema(value,json.loads(prepare_tools(TOOLS,choice)[1])),value
        return call
    reports=[]
    try:
        for alias,path in [('a',args.model),('b',args.other_model)]:
            runtime.manage('load',model=alias);model=runtime.runtime.host[alias]
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
            thinking={'enable_thinking':False} if model.config['model_type']=='qwen3' else {}
            for choice in ['required',{'type':'function','function':{'name':'translate'}}]:
                tools,_=prepare_tools(TOOLS,choice)
                rendered=tokenizer.apply_chat_template(MESSAGES,tools=tools,tokenize=False,
                    add_generation_prompt=True,**thinking)+'<tool_call>\n'
                assert model.chat_tokens(MESSAGES,tools=tools,tool_json=True,**thinking)==tokenizer.encode(rendered,add_special_tokens=False)
                result=request(alias,tool_choice=choice);call=check(result,choice)
                with post(alias,tool_choice=choice,stream=True,stream_options={'include_usage':True}) as response:
                    chunks=list(events(response, require_terminated=True))
                assert chunks[-1]=='[DONE]'
                payloads=[json.loads(c) for c in chunks[:-1]]
                streamed=[c for p in payloads for item in p['choices'] for c in item['delta'].get('tool_calls',[])]
                assert len(streamed)==1 and streamed[0]['function']==call['function'],streamed
                assert payloads[-1]['usage']==result['usage']
                assert payloads[-2]['choices'][0]['finish_reason']=='tool_calls'
                reports.append({'model':path,'choice':choice,'call':call['function'],
                    'usage':result['usage'],'timing':result['damendaemon_timing'],'stream_equal':True,'template_ids_equal':True})
            short=request(alias,max_tokens=1)['choices'][0]
            assert short['finish_reason']=='length' and not short['message'].get('tool_calls'),short
            assert short['message']['content'],short
        result=request('a');check(result);assert result['damendaemon_timing']['source_tier']=='host'
        barrier=threading.Barrier(4)
        def run(index):
            choice='required' if index%2==0 else {'type':'function','function':{'name':'translate'}}
            barrier.wait();result=request(tool_choice=choice);return result,check(result,choice)
        with ThreadPoolExecutor(4) as pool:concurrent=list(pool.map(run,range(4)))
        assert len({call['id'] for _,call in concurrent})==4
        max_batch=max(r['damendaemon_timing']['max_batch_size'] for r,_ in concurrent);assert max_batch>=2
        with post(stream=True) as response:next(events(response, require_terminated=True))
        check(request())
        auto=copy.deepcopy(TOOLS)
        for t in auto:t['function']['strict']=False
        with post(tools=auto,tool_choice='auto') as response:assert json.load(response)['choices']
        print(json.dumps({'passed':True,'device':args.device,'reports':reports,
            'max_batch':max_batch,'host_restore':True,'disconnect_recovery':True,'truncation_preserved':True},indent=2))
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()


if __name__=='__main__':main()
