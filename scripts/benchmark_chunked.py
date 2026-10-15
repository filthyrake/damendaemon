"""Measure active-stream stalls when a long prompt joins the same model group."""
from dllm.evidence_environment import report_environment
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.request import Request,urlopen
from tokenizers import Tokenizer
from dllm.bench import events



def check_usage(usage, prompt_tokens, completion_tokens):
    if (not isinstance(usage, dict)
            or type(usage.get('prompt_tokens')) is not int
            or type(usage.get('completion_tokens')) is not int
            or usage['prompt_tokens'] != prompt_tokens
            or usage['completion_tokens'] != completion_tokens):
        raise ValueError('benchmark requires exact integer prompt and completion usage')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--rounds',type=int,default=3)
    args=p.parse_args()
    if args.rounds < 1:p.error('rounds must be positive')
    tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'))
    tokenizer.no_padding();tokenizer.no_truncation()
    long_prompt=tokenizer.encode('The blue bird flies over the green field. '*200,add_special_tokens=True).ids[:1536]
    if len(long_prompt)!=1536:raise ValueError('benchmark requires a 1536-token long prompt')
    short_prompt=tokenizer.encode('Explain matrix multiplication in detail.',add_special_tokens=True).ids
    measurements=[]
    short_expected=long_expected=None
    for chunk in (0,128,256):
        with tempfile.TemporaryDirectory(prefix='dllm-chunk-bench-') as directory:
            logpath=Path(directory)/'server.log'
            with logpath.open('w') as log:
                process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--device',args.device,
                    '--context','2048','--batch-size','4','--prefill-chunk-size',str(chunk),
                    '--model','a='+args.model],stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+120
                    while time.monotonic()<deadline and process.poll() is None:
                        ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                        if ready:base=ready[-1].split()[-1];break
                        time.sleep(.1)
                    else:raise RuntimeError(logpath.read_text())
                    def request(prompt,count,stream=False):
                        body={'model':'a','prompt':prompt,'max_tokens':count,'stream':stream,'temperature':0}
                        if stream:body['stream_options']={'include_usage':True}
                        return urlopen(Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180)
                    def complete(prompt,count):
                        start=time.perf_counter()
                        with request(prompt,count) as response:result=json.load(response)
                        check_usage(result.get('usage'),len(prompt),count)
                        return result,time.perf_counter()-start
                    complete(long_prompt,8);complete(short_prompt,64)
                    trials=[]
                    for trial in range(args.rounds):
                        arrivals=[];text='';future=None;done=False;usage=None
                        start=time.perf_counter()
                        with ThreadPoolExecutor(1) as pool:
                            with request(short_prompt,64,True) as response:
                                for event in events(response, require_terminated=True):
                                    if event=='[DONE]':
                                        done=True;break
                                    data=json.loads(event)
                                    if 'error' in data:raise RuntimeError(data)
                                    if data.get('usage') is not None:
                                        check_usage(data['usage'],len(short_prompt),64)
                                        usage=data['usage']
                                    choices=data.get('choices',[])
                                    if (not isinstance(choices,list) or len(choices)>1
                                            or any(not isinstance(choice,dict) for choice in choices)):
                                        raise ValueError('benchmark requires one completion choice')
                                    delta=choices[0].get('text','') if choices else ''
                                    if not isinstance(delta,str):raise ValueError('benchmark output must be text')
                                    if delta:
                                        arrivals.append(time.perf_counter()-start)
                                        text+=delta
                                        if len(arrivals)==4:future=pool.submit(complete,long_prompt,8)
                            short_seconds=time.perf_counter()-start
                            if not done:raise ValueError('benchmark stream missing DONE')
                            check_usage(usage,len(short_prompt),64)
                            if future is None or len(arrivals)<5:
                                raise ValueError('benchmark requires overlap and a subsequent output sample')
                            long_result,long_seconds=future.result()
                        long_text=long_result['choices'][0]['text']
                        if short_expected is None:short_expected,long_expected=text,long_text
                        if text!=short_expected or long_text!=long_expected:
                            raise ValueError('chunking changed benchmark greedy output')
                        gaps=[b-a for a,b in zip(arrivals[3:],arrivals[4:])]
                        trials.append({'short_total_seconds':short_seconds,'long_total_seconds':long_seconds,
                            'max_active_stream_gap_ms':max(gaps)*1000,'short_content_chunks':len(arrivals),
                            'long_usage':long_result['usage'],'arrival_seconds':arrivals})
                    measurements.append({'chunk_size':chunk,'trials':trials,
                        'median_max_active_gap_ms':statistics.median(t['max_active_stream_gap_ms'] for t in trials),
                        'median_long_seconds':statistics.median(t['long_total_seconds'] for t in trials),
                        'median_short_seconds':statistics.median(t['short_total_seconds'] for t in trials)})
                except Exception:
                    print(logpath.read_text(),file=sys.stderr);raise
                finally:
                    process.terminate()
                    try:process.wait(30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
    print(json.dumps({'device':args.device,'model':args.model,'long_prompt_tokens':len(long_prompt),
        'runtime_environment':report_environment(os.environ),
        'short_prompt_tokens':len(short_prompt),'short_output':short_expected,'long_output':long_expected,
        'same_greedy_outputs':True,'results':measurements},indent=2))


if __name__=='__main__':main()
