"""Live JSON/schema proof through the real HTTP scheduler and tokenizer."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dllm.bench import events


def main():
    p=argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--device',default='cuda',choices=('cuda','mps'))
    p.add_argument('--raw',action='store_true')
    p.add_argument('--host-gib',default=8,type=float)
    args=p.parse_args()
    schema={'type':'object','properties':{'name':{'type':'string','const':'café'},
        'ok':{'type':'boolean','const':True},'values':{'type':'array','items':{'type':'integer','enum':[1,2]}}},
        'required':['name','ok','values'],'additionalProperties':False}
    fmt={'type':'json_schema','json_schema':{'name':'test','strict':True,'schema':schema}}
    prompt='Return only a JSON object with name café, ok true, and values an empty array.'
    with tempfile.TemporaryDirectory(prefix='dllm-structured-') as directory:
        logpath=Path(directory)/'server.log'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--context','512',
                '--device',args.device,'--host-gib',str(args.host_gib),'--model','a='+args.path],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline and process.poll() is None:
                    ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                    if ready:
                        base=ready[-1].split()[-1];break
                    time.sleep(.1)
                else:
                    raise RuntimeError(logpath.read_text())
                def post(raw=args.raw,**options):
                    body={'model':'a','max_tokens':96,'response_format':fmt,**options}
                    body.update({'prompt':prompt} if raw else {'messages':[{'role':'user','content':prompt}]})
                    return urlopen(Request(base+('/v1/completions' if raw else '/v1/chat/completions'),
                        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180)
                def request(**options):
                    with post(**options) as response:return json.load(response)
                def text(item):
                    c=item['choices'][0]
                    return c['text'] if 'text' in c else c['message']['content']
                def check(item):
                    assert item['choices'][0]['finish_reason']=='stop',item
                    value=json.loads(text(item))
                    assert set(value)=={'name','ok','values'} and value['name']=='café' and value['ok'] is True,value
                    assert isinstance(value['values'],list) and all(type(x) is int and x in (1,2) for x in value['values']),value
                first=request();check(first)
                raw=request(raw=True);check(raw)
                with post(stream=True,stream_options={'include_usage':True}) as response:chunks=list(events(response, require_terminated=True))
                assert chunks[-1]=='[DONE]'
                parsed=[json.loads(v) for v in chunks[:-1]]
                streamed=''.join(c.get('text',c.get('delta',{}).get('content','')) for item in parsed for c in item['choices'])
                assert streamed==text(first) and parsed[-1]['usage']==first['usage']
                plain=request(response_format={'type':'text'})
                with ThreadPoolExecutor(4) as pool:
                    mixed=list(pool.map(lambda i:request(response_format=fmt if i%2 else {'type':'text'}),range(4)))
                assert max(item["dllm_timing"]["max_batch_size"] for item in mixed)>=2
                for i,item in enumerate(mixed):
                    if i%2:check(item)
                    else:assert text(item)==text(plain)
                sampled=[request(temperature=.8,top_k=20,seed=17) for _ in range(2)]
                for item in sampled:check(item)
                assert text(sampled[0])==text(sampled[1])
                truncated=request(max_tokens=1)
                assert truncated['choices'][0]['finish_reason']=='length'
                obj=request(response_format={'type':'json_object'})
                assert obj['choices'][0]['finish_reason']=='stop' and isinstance(json.loads(text(obj)),dict)
                for options in ({'stop':'x'}, {'response_format':{'type':'json_schema','json_schema':{'name':'bad','schema':{'type':'string','pattern':'x'}}}}):
                    try:
                        post(**options).close();raise AssertionError('invalid request accepted')
                    except HTTPError as error:
                        assert error.code==400;error.close()
                # Disconnect an active constrained stream, then prove slot reuse.
                with post(stream=True,max_tokens=256) as response:
                    next(events(response, require_terminated=True))
                check(request())
                print(json.dumps({'passed':True,'model_path':args.path,'device':args.device,'raw':args.raw,
                    'mixed_batch':True,'stream_and_raw':True,'unicode':True,'sampling':True,
                    'truncation_and_recovery':True,'disconnect_recovery':True,'sample':first},indent=2))
            except Exception:
                print(logpath.read_text(),file=sys.stderr);raise
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
