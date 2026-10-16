"""Real Qwen2 Metal Q4/Q8 HTTP qualification against an independent CPU trace."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch
from tokenizers import Tokenizer
from dllm.bench import events


def cases_from_trace(trace):
    active={};cases=[]
    for event in trace['trace']:
        for row,slot in enumerate(event['slots']):
            token=event['targets'][row].argmax().item()
            if event['kind']=='prefill':
                case=dict(prompt=event['prompts'][row],tokens=[token])
                cases.append(case);active[slot]=case
            else:
                assert event['tokens'][row]==active[slot]['tokens'][-1]
                active[slot]['tokens'].append(token)
    return [dict(prompt=c['prompt'],tokens=c['tokens'][:8]) for c in cases if len(c['tokens'])>=4]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('trace');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    cases=cases_from_trace(trace);assert len(cases)>=4
    model=trace['path'];tokenizer=Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
    eos=json.loads((Path(model)/'config.json').read_text())['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos
    for case in cases:
        for index,token in enumerate(case['tokens']):
            if token in eos:case['tokens']=case['tokens'][:index+1];break
    report=dict(passed=False,model=model,quantization=trace.get('reference_quantization','none'),
        trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),checks=[],trials=[],statuses={},
        scope='Metal quantized HTTP and Q4/Q8 precision switching')
    with tempfile.TemporaryDirectory(prefix='dllm-metal-q4-http-') as directory:
        logpath=Path(directory)/'server.log'
        env={**os.environ,'DLLM_QUANTIZATION':report['quantization'],'DLLM_DEVICE':'mps','DLLM_METAL_WEIGHT_DTYPE':'float16'}
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--device','mps',
                '--context','2048','--batch-size','4','--batch-wait-ms','10','--gpu-models','1',
                '--host-models','2','--host-gib','8'],stdout=log,stderr=log,env=env)
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline and process.poll() is None:
                    lines=[line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                    if lines:base=lines[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                def opened(path,body=None):
                    return urlopen(Request(base+path,data=None if body is None else json.dumps(body).encode(),
                        headers={'Content-Type':'application/json'}),timeout=180)
                def call(path,body=None):
                    with opened(path,body) as response:return json.load(response)
                def body(index=0,alias='a',count=None):
                    case=cases[index]
                    return dict(model=alias,prompt=case['prompt'],temperature=0,max_tokens=count or len(case['tokens']))
                def request(index=0,alias='a',count=None):
                    started=time.perf_counter();result=call('/v1/completions',body(index,alias,count))
                    result['wall_seconds']=time.perf_counter()-started;return result
                def check(result,index=0,count=None,label='reference'):
                    case=cases[index];tokens=case['tokens'][:count]
                    assert result['choices'][0]['text']==tokenizer.decode(tokens,skip_special_tokens=True),label
                    assert result['usage']['prompt_tokens']==len(case['prompt']),label
                    assert result['usage']['completion_tokens']==len(tokens),label
                    report['checks'].append(label)
                    report['trials'].append(dict(label=label,wall_seconds=result['wall_seconds'],timing=result['dllm_timing']))
                def status(label):
                    value=call('/v1/models/status');report['statuses'][label]=value;return value
                assert status('initial')['models']==[]
                assert report['quantization'] in ('q4_0','q8_0')
                other='q8_0' if report['quantization']=='q4_0' else 'q4_0'
                for alias,quantization in (('a',report['quantization']),('b',other)):
                    call('/v1/models/register',dict(model=alias,path=model,quantization=quantization))
                for index in range(len(cases)):check(request(index),index,label='independent:'+str(index))
                loaded=status('loaded');assert loaded['device']=='mps'
                counts=(2,8,4,6)
                with ThreadPoolExecutor(4) as pool:
                    responses=list(pool.map(lambda pair:request(pair[0],count=pair[1]),enumerate(counts)))
                for index,result in enumerate(responses):check(result,index,counts[index],'concurrent-shrinking:'+str(index))
                assert max(r['dllm_timing']['max_batch_size'] for r in responses)>=2
                with opened('/v1/completions',{**body(),'stream':True,'stream_options':{'include_usage':True}}) as response:
                    chunks=list(events(response, require_terminated=True))
                assert chunks[-1]=='[DONE]'
                parsed=[json.loads(c) for c in chunks[:-1]]
                text=''.join(c['text'] for chunk in parsed for c in chunk.get('choices',[]))
                assert text==tokenizer.decode(cases[0]['tokens'],skip_special_tokens=True)
                assert parsed[-1]['usage']['completion_tokens']==len(cases[0]['tokens'])
                report['checks'].append('streaming-content-and-usage')
                with opened('/v1/completions',{**body(),'stream':True,'max_tokens':128}) as response:
                    for event in events(response, require_terminated=True):
                        if event=='[DONE]':raise AssertionError('stream ended before cancellation')
                        if any(c.get('text') for c in json.loads(event).get('choices',[])):break
                check(request(1),1,label='cancel-and-reuse-slot')
                for invalid in ({**body(),'max_tokens':3000},{**body(),'prompt':[151936]}):
                    try:call('/v1/completions',invalid)
                    except HTTPError as error:
                        with error:
                            if error.code != 400:
                                raise ValueError('unexpected HTTP rejection status')
                    else:raise AssertionError('invalid request accepted')
                check(request(),label='invalid-request-recovery')
                chat=dict(model='a',messages=[dict(role='user',content='Explain binary search briefly.')],max_tokens=8)
                scored=call('/v1/chat/completions',{**chat,'logprobs':True,'top_logprobs':2})
                scores=scored['choices'][0]['logprobs']['content']
                assert scores and all(math.isfinite(s['logprob']) and len(s['top_logprobs'])==2 for s in scores)
                report['checks'].append('native-logprobs')
                sampled={**chat,'temperature':.7,'top_k':20,'top_p':.9,'seed':42,'repetition_penalty':1.1}
                assert call('/v1/chat/completions',sampled)['choices']==call('/v1/chat/completions',sampled)['choices']
                report['checks'].append('seeded-sampling-and-penalties')
                other_result=request(alias='b')
                assert other_result['choices'][0]['text'] and other_result['usage']['completion_tokens']>0
                switched=status('other-precision-loaded')
                assert next(m for m in switched['models'] if m['id']=='b')['quantization']==other
                report['checks'].append('other-precision-switch-smoke')
                restored=request();check(restored,label='host-restore')
                assert restored['dllm_timing']['source_tier']=='host'
                for alias in ('a','b'):call('/v1/models/unload',dict(model=alias,release_host=True))
                unloaded=status('unloaded')
                assert unloaded['host_cached_bytes']==0 and all(m['partition'] is None for m in unloaded['models'])
                check(request(),label='disk-reload-after-release')
                call('/v1/models/unload',dict(model='a',release_host=True))
                again=status('unloaded-again')
                assert again['host_cached_bytes']==0
                assert again['mps_allocated_bytes']==unloaded['mps_allocated_bytes']
                report['checks'].append('repeat-release-no-metal-allocation-growth')
                report['passed']=True
            except BaseException as error:
                report['error']=repr(error);print(logpath.read_text(),file=sys.stderr);raise
            finally:
                process.terminate()
                try:process.wait(30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']))))


if __name__=='__main__':main()
