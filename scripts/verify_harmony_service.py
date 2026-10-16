"""Native GPT-OSS text Harmony HTTP channels and request lifecycle on A2."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import weakref

import torch
from dllm.scheduler import BatchedRuntime
from dllm.server import BoundedHTTPServer,handler_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('swap_model')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':raise RuntimeError('restricted to A2')
    os.environ['DLLM_DEVICE']='cuda';torch.set_num_threads(4)
    runtime=BatchedRuntime({'oss':args.model,'swap':args.swap_model},context=512,batch_size=2,host_gib=18,host_models=2,gpu_models=1)
    server=BoundedHTTPServer(('127.0.0.1',0),handler_for(runtime));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    base={'model':'oss','messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
        'max_tokens':128,'temperature':0,'reasoning_effort':'low'}
    rows=[]
    def request(body):return urlopen(Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=240)
    def post(body):
        with request(body) as response:return json.load(response)
    def equal(a,b):
        assert a['choices']==b['choices'];assert a['usage']==b['usage']
    def stream(body,disconnect=False):
        content='';reasoning='';finish=None;usage=None;done=False
        with request({**body,'stream':True,'stream_options':{'include_usage':True}}) as response:
            for raw in response:
                if not raw.startswith(b'data: '):continue
                line=raw[6:].strip()
                if line==b'[DONE]':done=True;break
                event=json.loads(line);assert 'error' not in event,event
                if 'usage' in event:usage=event['usage']
                for choice in event['choices']:
                    delta=choice.get('delta',{})
                    content+=delta.get('content','');reasoning+=delta.get('reasoning_content','')
                    if choice.get('finish_reason'):finish=choice['finish_reason']
                if disconnect and reasoning:break
        if not disconnect:assert done
        return content,reasoning,finish,usage
    try:
        histories=[base['messages'],
            [{'role':'user','content':'My name is Ada.'},{'role':'assistant','content':'Hello, Ada.'},
             {'role':'user','content':'What is my name? Answer briefly.'}],
            [{'role':'user','content':'Reply with exactly this text: café 🐈'}]]
        for index,messages in enumerate(histories):
            body={**base,'messages':messages};result=post(body);choice=result['choices'][0];message=choice['message']
            actual=stream(body)
            assert actual==(message['content'],message.get('reasoning_content',''),choice['finish_reason'],result['usage'])
            assert choice['finish_reason']=='stop' and message['content']
            assert '<|' not in message['content'] and '<|' not in message.get('reasoning_content','')
            if index==0:assert message['content'].strip()=='4'
            if index==1:assert 'Ada' in message['content']
            if index==2:assert 'café' in message['content'] and '🐈' in message['content']
            rows.append({'case':'chat_stream_equivalence','response':result})
        reference=rows[0]['response']
        echoed={**base,'messages':base['messages']+[reference['choices'][0]['message'],
            {'role':'user','content':'Now add one to that result. Answer briefly.'}]}
        echo_result=post(echoed)
        assert '5' in echo_result['choices'][0]['message']['content']
        rows.append({'case':'response_message_history_roundtrip','response':echo_result})
        for cap in (1,2,3,4):
            body={**base,'max_tokens':cap};result=post(body);message=result['choices'][0]['message']
            assert result['choices'][0]['finish_reason']=='length' and result['usage']['completion_tokens']==cap
            assert stream(body)==(message['content'],message.get('reasoning_content',''),'length',result['usage'])
            assert message['content']==''
            rows.append({'case':'header_or_analysis_token_cap','cap':cap,'response':result})
        stopped=post({**base,'stop':'4'})
        assert stopped['choices'][0]['message']['content']==''
        assert stopped['choices'][0]['message']['reasoning_content']==reference['choices'][0]['message']['reasoning_content']
        assert stopped['choices'][0]['finish_reason']=='stop'
        rows.append({'case':'final_content_stop','response':stopped})
        stream(base,disconnect=True);equal(post(base),reference)
        with ThreadPoolExecutor(2) as pool:concurrent=list(pool.map(lambda _:post(base),range(2)))
        for result in concurrent:equal(result,reference)
        assert max(result['dllm_timing']['max_batch_size'] for result in concurrent)==2
        for invalid in ({**base,'logprobs':True},{**base,'response_format':{'type':'json_object'}},
            {**base,'tool_choice':'required','tools':[{'type':'function','function':{'name':'test','parameters':{'type':'object'}}}]},
            {**base,'reasoning_effort':'maximum'},{**base,'chat_template_kwargs':{'enable_thinking':True}}):
            try:post(invalid)
            except HTTPError as error:
                with error:
                    if error.code != 400:
                        raise ValueError('unexpected HTTP rejection status')
            else:raise AssertionError('unsupported Harmony request accepted')
        with runtime.runtime.session('oss') as (executor,_,_):
            old=weakref.ref(executor);graph=weakref.ref(executor.batch_graphs[1][4])
        del executor
        post({'model':'swap','messages':[{'role':'user','content':'Say hello.'}],'max_tokens':2,'temperature':0})
        gc.collect();torch.cuda.synchronize();assert old() is None and graph() is None
        restored=post(base);equal(restored,reference);assert restored['dllm_timing']['source_tier']=='host'
        result={'passed':True,'gpu':torch.cuda.get_device_name(),'context':512,'batch_size':2,'rows':rows,
            'disconnect_during_analysis_recovery':True,'concurrent_channel_isolation':True,
            'unsupported_forced_tools_schema_scores_rejected':True,'swap_graph_reclaimed':True,'host_restore_equivalent':True}
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'output':args.output}))
    finally:
        server.shutdown();server.server_close();thread.join();runtime.close()


if __name__=='__main__':main()
