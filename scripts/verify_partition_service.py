"""Real HTTP qualification of same-host two-GPU residency and request lifetime."""
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
    return [dict(prompt=c['prompt'],tokens=c['tokens'][:8]) for c in cases if len(c['tokens'])>=8]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('trace');parser.add_argument('--output',required=True)
    parser.add_argument('--inject-stage-failure',action='store_true')
    parser.add_argument('--experimental-q4-body',nargs='?',const='single-dot',choices=('single-dot','paired-load','adjacent-gate','adjacent-all','adjacent-inplace','stage-dynamic'))
    parser.add_argument('--allocation-diagnostics',action='store_true');args=parser.parse_args()
    if args.inject_stage_failure and args.allocation_diagnostics:parser.error('choose one diagnostic mode')
    if args.experimental_q4_body and (args.inject_stage_failure or args.allocation_diagnostics):
        parser.error('Q4 body experiment cannot combine server overrides')
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    if args.experimental_q4_body and trace.get('reference_quantization')!='q4_0':
        parser.error('Q4 body experiment requires a Q4 reference')
    cases=cases_from_trace(trace);assert len(cases)>=4
    model=trace['path'];tokenizer=Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
    eos=json.loads((Path(model)/'config.json').read_text())['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos
    for case in cases:
        for index,token in enumerate(case['tokens']):
            if token in eos:case['tokens']=case['tokens'][:index+1];break
    report=dict(passed=False,model=model,quantization=trace.get('reference_quantization','none'),
        trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),checks=[],trials=[],statuses={},
        injected_stage_failure=args.inject_stage_failure,allocation_diagnostics=args.allocation_diagnostics,
        experimental_q4_body=args.experimental_q4_body,scope='same-host two-GPU HTTP qualification')
    with tempfile.TemporaryDirectory(prefix='dllm-partition-http-') as directory:
        logpath=Path(directory)/'server.log'
        env={**os.environ,'DLLM_QUANTIZATION':report['quantization'],'DLLM_DEVICE':'cuda'}
        with logpath.open('w') as log:
            command=([sys.executable,__file__,'--failure-server'] if args.inject_stage_failure else
                     [sys.executable,__file__,'--diagnostic-server'] if args.allocation_diagnostics else
                     [sys.executable,'-m','scripts.experiments.run_q4_body','--body-kernel',args.experimental_q4_body] if args.experimental_q4_body else
                     [sys.executable,'-m','dllm'])
            process=subprocess.Popen([*command,'serve','--port','0','--context','512','--layer-split','12',
                '--batch-size','4','--batch-wait-ms','10','--gpu-models','1','--host-models','2','--host-gib','8',
                '--model','a='+model,'--model','b='+model],stdout=log,stderr=log,env=env)
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
                status('initial')
                if args.inject_stage_failure:
                    try:request()
                    except HTTPError as error:
                        with error:
                            if error.code != 500:
                                raise ValueError('unexpected HTTP rejection status')
                    else:raise AssertionError('injected stage failure returned success')
                    failed=status('failed-executor')
                    assert failed['models'][0]['executor_failed']
                    result=request();check(result,label='reactivation-after-partial-stage-failure')
                    assert result['dllm_timing']['source_tier']=='host'
                    assert not status('reactivated')['models'][0]['executor_failed']
                for index in range(len(cases)):check(request(index),index,label='independent:'+str(index))
                loaded=status('loaded');assert len(loaded['cuda_devices'])==2
                assert [s['layers'] for s in loaded['models'][0]['partition']]==[[0,12],[12,24]]
                assert all(s['kv_bytes']==12582912 for s in loaded['models'][0]['partition'])
                if not args.experimental_q4_body:
                    from dllm.partition_policy import q4_stage_adjacent_enabled
                    layout='q4-adjacent-body' if q4_stage_adjacent_enabled(report['quantization']) else 'canonical'
                    assert all(s['weight_layout']==layout for s in loaded['models'][0]['partition'])
                    if layout=='q4-adjacent-body':
                        from dllm.q4_stage import runtime_rows_enabled
                        assert all(s['runtime_row_bounds']==runtime_rows_enabled() for s in loaded['models'][0]['partition'])
                        from dllm.partition_policy import q4_stage_runtime_rope_enabled
                        assert all(s['runtime_rope_tokens']==q4_stage_runtime_rope_enabled(report['quantization']) for s in loaded['models'][0]['partition'])
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
                for invalid in ({**body(),'max_tokens':600},{**body(),'prompt':[151936]}):
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
                report['checks'].append('native-logprobs-on-output-gpu')
                sampled={**chat,'temperature':.7,'top_k':20,'top_p':.9,'seed':42,'repetition_penalty':1.1}
                assert call('/v1/chat/completions',sampled)['choices']==call('/v1/chat/completions',sampled)['choices']
                report['checks'].append('seeded-sampling-and-penalties')
                check(request(alias='b'),label='second-alias')
                restored=request();check(restored,label='host-restore')
                assert restored['dllm_timing']['source_tier']=='host'
                for alias in ('a','b'):call('/v1/models/unload',dict(model=alias,release_host=True))
                unloaded=status('unloaded')
                assert unloaded['host_cached_bytes']==0 and all(m['partition'] is None for m in unloaded['models'])
                check(request(),label='disk-reload-after-release')
                call('/v1/models/unload',dict(model='a',release_host=True))
                again=status('unloaded-again')
                assert again['host_cached_bytes']==0
                assert [d['allocated_bytes'] for d in again['cuda_devices']]==[d['allocated_bytes'] for d in unloaded['cuda_devices']]
                report['checks'].append('repeat-release-no-per-device-allocation-growth')
                if args.allocation_diagnostics:
                    for value in (unloaded,again):
                        blocks=value['diagnostic_retained_allocations']
                        assert all(block['cublas_workspace'] for block in blocks),blocks
                        for device in value['cuda_devices']:
                            assert device['allocated_bytes']==sum(b['bytes'] for b in blocks if b['device']==device['index'])
                    report['checks'].append('retained-allocations-identified-as-cublas-workspaces')
                report['passed']=True
            except BaseException as error:
                report['error']=repr(error);print(logpath.read_text(),file=sys.stderr);raise
            finally:
                process.terminate()
                try:process.wait(30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],checks=len(report['checks']))))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--failure-server':
        from dllm.layer_stage import LayerStage
        original=LayerStage.run_decode
        state={'failed':False}
        def fail_once(stage,*args,**kwargs):
            if stage.last and not state['failed']:
                state['failed']=True
                raise RuntimeError('injected last-stage failure after first-stage decode')
            return original(stage,*args,**kwargs)
        LayerStage.run_decode=fail_once
        del sys.argv[1]
        from dllm.cli import main as serve
        serve()
    elif len(sys.argv)>1 and sys.argv[1]=='--diagnostic-server':
        from dllm.runtime import Runtime
        torch.cuda.memory._record_memory_history(max_entries=20000)
        original_status=Runtime._status
        def diagnostic_status(runtime):
            result=original_status(runtime)
            if result['host_cached_bytes']==0:
                snapshot=torch.cuda.memory._snapshot()
                result['diagnostic_retained_allocations']=[dict(device=segment['device'],bytes=block['size'],
                    cublas_workspace=any('getNewWorkspace' in f.get('name','') for f in block.get('frames',[])),
                    frame_names=[f.get('name','') for f in block.get('frames',[])])
                    for segment in snapshot['segments'] for block in segment['blocks'] if block['state']=='active_allocated']
            return result
        Runtime._status=diagnostic_status
        del sys.argv[1]
        from dllm.cli import main as serve
        serve()
    else:main()
