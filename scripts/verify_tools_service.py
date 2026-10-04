"""Real GPU function-call generation, history, streaming and isolation proof."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
from urllib.request import Request,urlopen
import torch
from transformers import AutoTokenizer
from dllm.bench import events
from dllm.runtime import require_gpu
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for
from dllm.tool_calls import template_messages

TOOLS=[{'type':'function','function':{'name':'get_weather',
    'description':'Get current weather for a city. Use this for current conditions.',
    'parameters':{'type':'object','properties':{'city':{'type':'string'},
        'unit':{'type':'string','enum':['celsius','fahrenheit']}},'required':['city','unit']}}}]
MESSAGES=[{'role':'user','content':"What is the current weather in Paris? Use get_weather with city='Paris' and unit='celsius'."}]


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--other-model',required=True)
    p.add_argument('--thinking',action='store_true')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda');args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device;require_gpu();torch.set_num_threads(4)
    paths={'a':args.model,'b':args.other_model}
    runtime=BatchedRuntime(paths,context=2048,batch_size=4,gpu_models=1,host_models=2,host_gib=16)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    thinking={alias:({'enable_thinking':args.thinking} if json.load(open(path+'/config.json'))['model_type']=='qwen3' else {})
              for alias,path in paths.items()}
    def post(alias='a',stream=False,**changes):
        body={'model':alias,'messages':MESSAGES,'tools':TOOLS,'temperature':0,'max_tokens':512 if args.thinking else 128,
              'chat_template_kwargs':thinking[alias],'stream':stream,**changes}
        if stream:body['stream_options']={'include_usage':True}
        return urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
            headers={'Content-Type':'application/json'}),timeout=180)
    def request(alias='a',**changes):
        with post(alias,**changes) as response:return json.load(response)
    def check(result):
        choice=result['choices'][0]
        assert choice['finish_reason']=='tool_calls',result
        calls=choice['message']['tool_calls'];assert len(calls)==1,calls
        call=calls[0];assert call['function']['name']=='get_weather',call
        arguments=json.loads(call['function']['arguments'])
        assert arguments=={'city':'Paris','unit':'celsius'},arguments
        return call
    reports=[];baseline={}
    try:
        for alias,path in paths.items():
            runtime.manage('load',model=alias);model=runtime.runtime.host[alias]
            tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
            actual=model.chat_tokens(MESSAGES,tools=TOOLS,**thinking[alias])
            expected=tokenizer.apply_chat_template(MESSAGES,tools=TOOLS,add_generation_prompt=True,return_dict=False,**thinking[alias])
            assert actual==expected,'function definition template mismatch'
            whole=request(alias);call=check(whole);baseline[alias]=call['function']
            with post(alias,stream=True) as response:chunks=list(events(response))
            assert chunks[-1]=='[DONE]'
            payloads=[json.loads(x) for x in chunks[:-1]];calls={};content=''
            for payload in payloads:
                for choice in payload['choices']:
                    delta=choice['delta'];content+=delta.get('content','')
                    for item in delta.get('tool_calls',[]):
                        current=calls.setdefault(item['index'],{'id':'','name':'','arguments':''})
                        current['id']+=item.get('id','')
                        current['name']+=item.get('function',{}).get('name','')
                        current['arguments']+=item.get('function',{}).get('arguments','')
            assert list(calls)==[0] and calls[0]['id']!=call['id']
            assert calls[0]['name']==call['function']['name']
            assert json.loads(calls[0]['arguments'])==json.loads(call['function']['arguments'])
            assert content==(whole['choices'][0]['message']['content'] or '')
            assert payloads[-1]['usage']==whole['usage']
            assert payloads[-2]['choices'][0]['finish_reason']=='tool_calls'
            history=MESSAGES+[whole['choices'][0]['message'],{'role':'tool','tool_call_id':call['id'],
                'content':json.dumps({'city':'Paris','temperature':21,'unit':'celsius','conditions':'sunny'})}]
            expected=tokenizer.apply_chat_template(template_messages(history),tools=TOOLS,
                add_generation_prompt=True,return_dict=False,**thinking[alias])
            assert model.chat_tokens(history,tools=TOOLS,**thinking[alias])==expected,'tool result template mismatch'
            continuation=request(alias,messages=history)
            assert not continuation['choices'][0]['message'].get('tool_calls'),continuation
            assert continuation['choices'][0]['finish_reason']=='stop',continuation
            assert '21' in continuation['choices'][0]['message']['content'].split('</think>')[-1],continuation
            suppressed=request(alias,tool_choice='none')
            assert not suppressed['choices'][0]['message'].get('tool_calls')
            reports.append({'model':path,'template_ids_equal':True,'history_template_ids_equal':True,
                'tool_call':call['function'],'continuation':continuation['choices'][0],
                'stream_and_nonstream_equal':True,'tool_choice_none':True})
        restored=request('a');check(restored)
        assert restored['dllm_timing']['source_tier']=='host'
        barrier=threading.Barrier(4)
        def concurrent(_):barrier.wait();return request('a')
        with ThreadPoolExecutor(4) as pool:outputs=list(pool.map(concurrent,range(4)))
        ids=[check(result)['id'] for result in outputs];assert len(set(ids))==4
        assert all(check(result)['function']==baseline['a'] for result in outputs)
        max_batch=max(r['dllm_timing']['max_batch_size'] for r in outputs);assert max_batch>=2,max_batch
        with post('a',stream=True) as response:next(events(response))
        runtime.manage('status');check(request('a'))
        print(json.dumps({'passed':True,'device':args.device,'reports':reports,'concurrent_calls':4,
            'qwen3_thinking':args.thinking,
            'runtime_environment':{k:v for k,v in os.environ.items() if k.startswith('DLLM_')},
            'max_batch_seen':max_batch,'request_local_ids':True,'host_restore':True,
            'disconnect_recovery':True,'tools_executed':False},indent=2))
    finally:
        server.shutdown();server.server_close();worker.join();runtime.close()


if __name__=='__main__':main()
