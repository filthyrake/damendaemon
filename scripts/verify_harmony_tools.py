"""Native GPT-OSS auto-function HTTP round trips, framing and lifecycle checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch
from dllm.bench import events
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for

TOOLS=[{'type':'function','function':{'name':'weather','description':'Get current weather for a city.',
    'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
MESSAGES=[{'role':'user','content':'Use the weather tool to get the current weather in Paris.'}]


def require_gate(condition, message):
    if not condition:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--output',required=True)
    p.add_argument('--context',type=int,default=1024);p.add_argument('--prefix-tokens',type=int,default=0);args=p.parse_args()
    if args.prefix_tokens<0:raise ValueError('prefix length must be nonnegative')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    messages=MESSAGES
    if args.prefix_tokens:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'))
        corpus=tokenizer.encode('Background: the ocean affects coastal weather through temperature and wind. ',add_special_tokens=False).ids
        ids=(corpus*((args.prefix_tokens+len(corpus)-1)//len(corpus)))[:args.prefix_tokens]
        messages=[{'role':'user','content':tokenizer.decode(ids)+'\n\n'+MESSAGES[0]['content']}]
    runtime=BatchedRuntime({'oss':args.model},context=args.context,batch_size=2,host_gib=18)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    base={'model':'oss','messages':messages,'max_tokens':128,'temperature':0,'reasoning_effort':'low','tools':TOOLS}
    rows=[]
    def request(body):return urlopen(Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=240)
    def post(body):
        with request(body) as response:return json.load(response)
    def stream(body,disconnect=False):
        result={'content':'','reasoning_content':''};calls=[];finish=None;usage=None;done=False
        with request({**body,'stream':True,'stream_options':{'include_usage':True}}) as response:
            for line in events(response,require_terminated=True):
                if line=='[DONE]':done=True;break
                event=json.loads(line);require_gate('error' not in event, 'stream returned an error')
                if 'usage' in event:usage=event['usage']
                for choice in event['choices']:
                    delta=choice.get('delta',{})
                    for key in result:result[key]+=delta.get(key,'')
                    for call in delta.get('tool_calls',[]):
                        require_gate(call.pop('index')==len(calls), 'unexpected streamed tool-call index');calls.append(call)
                    if choice.get('finish_reason'):finish=choice['finish_reason']
                if disconnect:break
        if not disconnect:require_gate(done, 'stream ended without DONE')
        return result,calls,finish,usage
    def canonical(response):
        message=response['choices'][0]['message']
        return (message.get('content') or '',message.get('reasoning_content',''),
            [c['function'] for c in message.get('tool_calls',[])],response['choices'][0]['finish_reason'],response['usage'])
    try:
        first=post(base);print(json.dumps(first),flush=True)
        choice=first['choices'][0];message=choice['message'];call=message['tool_calls'][0]
        require_gate(choice['finish_reason']=='tool_calls' and call['function']['name']=='weather', 'expected a weather tool call')
        require_gate(json.loads(call['function']['arguments'])=={'city':'Paris'}, 'unexpected weather tool arguments')
        actual,calls,finish,usage=stream(base)
        require_gate((actual['content'],actual['reasoning_content'],[c['function'] for c in calls],finish,usage)==canonical(first), 'stream differs from ordinary response')
        rows.append({'case':'auto_stream_equivalence','response':first})
        history=messages+[message,{'role':'tool','tool_call_id':call['id'],'content':'Paris: 19 degrees Celsius, sunny.'}]
        reply=post({**base,'messages':history});print(json.dumps(reply),flush=True)
        require_gate(reply['choices'][0]['finish_reason']=='stop' and '19' in reply['choices'][0]['message']['content'], 'follow-up did not incorporate the tool result')
        rows.append({'case':'tool_result_history','response':reply})
        for cap in (1,4,first['usage']['completion_tokens']-1):
            limited={**base,'max_tokens':cap};result=post(limited)
            require_gate(result['choices'][0]['finish_reason']=='length' and 'tool_calls' not in result['choices'][0]['message'], 'truncated call must finish by length without a tool call')
            actual,calls,finish,usage=stream(limited)
            require_gate((actual['content'],actual['reasoning_content'],[c['function'] for c in calls],finish,usage)==canonical(result), 'truncated stream differs from ordinary response')
            rows.append({'case':'incomplete_header_or_arguments','cap':cap,'response':result})
        none=post({**base,'tool_choice':'none','messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}]})
        require_gate(none['choices'][0]['finish_reason']=='stop' and none['choices'][0]['message']['content'].strip()=='4', 'tool_choice none did not produce the expected plain answer')
        rows.append({'case':'none_choice','response':none})
        stream(base,disconnect=True);require_gate(canonical(post(base))==canonical(first), 'generation changed after disconnect')
        with ThreadPoolExecutor(2) as pool:concurrent=list(pool.map(lambda _:post(base),range(2)))
        require_gate(all(canonical(r)==canonical(first) for r in concurrent), 'concurrent response differs from baseline')
        require_gate(concurrent[0]['choices'][0]['message']['tool_calls'][0]['id']!=concurrent[1]['choices'][0]['message']['tool_calls'][0]['id'], 'concurrent calls reused an ID')
        for invalid in ({**base,'tool_choice':'required'},
            {**base,'tool_choice':{'type':'function','function':{'name':'weather'}}},
            {**base,'tools':[{'type':'function','function':{**TOOLS[0]['function'],'strict':True}}]}):
            try:post(invalid)
            except HTTPError as error:
                with error:require_gate(error.code==400, 'unsupported choice must return HTTP 400')
            else:raise ValueError('unsupported choice accepted')
        captured=list(runtime.generate('oss',messages,128,tools=TOOLS,reasoning_effort='low'))
        rows.append({'case':'captured_tokens','tokens':[e['token'] for e in captured if 'token' in e],
            'content':''.join(e.get('text','') for e in captured),'reasoning_content':''.join(e.get('reasoning_content','') for e in captured),
            'calls':[c for e in captured for c in e.get('tool_calls',[])],'final':captured[-1]})
        result={'passed':True,'gpu':torch.cuda.get_device_name(),'context':args.context,'prefix_tokens':args.prefix_tokens,'prompt_messages':messages,'batch_size':2,'rows':rows,
            'disconnect_recovery':True,'concurrent_calls_isolated':True,'unsupported_choices_rejected':True}
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(rows)}))
    finally:server.shutdown();server.server_close();runtime.close()


if __name__=='__main__':main()
