"""Independent real-GPU HTTP proof for a resident two-rank service.

Default mode owns a bounded same-host torchrun process group. For cross-host
proof, launch this file with torchrun --run-path and --worker on each host;
only rank zero reads the reference and writes the report. Use an outer deadline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('model');p.add_argument('trace');p.add_argument('--output',required=True)
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),required=True)
    p.add_argument('--idle-seconds',type=float,default=35)
    p.add_argument('--batch-size',type=int,choices=range(1,5),default=1)
    p.add_argument('--worker',action='store_true')
    return p


def worker(args):
    from dllm.distributed_server_cli import main as serve
    if os.environ['RANK']!='0':
        serve([args.model,'--quantization',args.quantization,'--alias','a','--port','0','--batch-size',str(args.batch_size)]);return
    import torch
    from tokenizers import Tokenizer
    from dllm.bench import events
    from dllm.distributed_serving import DistributedRuntime
    from dllm.server import BoundedHTTPServer
    from scripts.verify_partition_service import cases_from_trace
    torch.set_num_threads(4)
    trace=torch.load(args.trace,map_location='cpu',weights_only=True)
    if trace.get('reference_quantization','none')!=args.quantization:raise ValueError('reference precision differs')
    cases=cases_from_trace(trace);assert len(cases)>=4
    tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'))
    eos=json.loads((Path(args.model)/'config.json').read_text())['eos_token_id']
    eos={eos} if isinstance(eos,int) else set(eos)
    for case in cases:
        for index,token in enumerate(case['tokens']):
            if token in eos:case['tokens']=case['tokens'][:index+1];break
    report=dict(passed=False,batch_size=args.batch_size,quantization=args.quantization,model=args.model,checks=[],trials=[],statuses={},
        trace_sha256=hashlib.sha256(Path(args.trace).read_bytes()).hexdigest(),
        coordinator_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),idle_seconds=args.idle_seconds,
        scope='Real two-rank greedy HTTP qualification; not a performance comparison',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('dllm/distributed.py','dllm/distributed_serving.py','dllm/distributed_server_cli.py','dllm/stage_protocol.py','dllm/layer_stage.py','dllm/q4_stage.py','dllm/q4_stage_kernels.py')})
    base=None
    original_init=BoundedHTTPServer.__init__
    original_run=DistributedRuntime.run
    def capture(server,*positional,**keywords):
        nonlocal base
        original_init(server,*positional,**keywords)
        base=f'http://127.0.0.1:{server.server_port}'
    def qualify(runtime):
        def opened(path,body=None):
            return urlopen(Request(base+path,data=None if body is None else json.dumps(body).encode(),
                headers={'Content-Type':'application/json'}),timeout=45)
        def call(path,body=None):
            with opened(path,body) as response:return json.load(response)
        def status(label):
            result=call('/v1/models/status');report['statuses'][label]=result;return result
        def body(index=0,count=None):
            case=cases[index]
            return dict(model='a',prompt=case['prompt'],temperature=0,max_tokens=count or len(case['tokens']))
        def request(index=0,count=None):return call('/v1/completions',body(index,count))
        def check(result,index=0,count=None,label='reference'):
            case=cases[index];tokens=case['tokens'][:count]
            assert result['choices'][0]['text']==tokenizer.decode(tokens,skip_special_tokens=True),label
            assert result['usage']['prompt_tokens']==len(case['prompt']),label
            assert result['usage']['completion_tokens']==len(tokens),label
            assert 1<=result['dllm_timing']['max_batch_size']<=args.batch_size,label
            assert 1<=result['dllm_timing']['prefill_batch_size']<=args.batch_size,label
            report['checks'].append(label);report['trials'].append(dict(label=label,timing=result['dllm_timing']))
        try:
            assert call('/health')['status']=='ok'
            assert [m['id'] for m in call('/v1/models')['data']]==['a']
            initial=status('initial');assert initial['distributed'] and initial['models'][0]['quantization']==args.quantization
            expected_layout='q4-adjacent-body' if runtime.pair.q4_stage_adjacent else 'canonical'
            assert initial['weight_layout']==expected_layout
            counts=(2,8,4,6)
            with ThreadPoolExecutor(4) as pool:
                initial_results=list(pool.map(lambda count:request(0,count),counts))
            for index,result in enumerate(initial_results):
                check(result,0,counts[index],'initial-queued-common-prefix:'+str(index))
            if initial['packed_prefill']:
                assert max(result['dllm_timing']['prefill_batch_size'] for result in initial_results)==args.batch_size
            for index in range(len(cases)):check(request(index),index,label='independent:'+str(index))
            for index in (1,0,2,0):check(request(index),index,label='resident-reuse:'+str(index))
            counts=(2,8,4,6)
            with ThreadPoolExecutor(4) as pool:
                results=list(pool.map(lambda pair:request(pair[0],pair[1]),enumerate(counts)))
            for index,result in enumerate(results):check(result,index,counts[index],'queued-concurrent:'+str(index))
            if args.batch_size>1:
                assert max(result['dllm_timing']['max_batch_size'] for result in results)>1,'no decode batching observed'
            if args.batch_size>1:
                lengths=(2,8,4,6,3,5)
                with ThreadPoolExecutor(6) as pool:
                    refills=list(pool.map(lambda item:request(item[0]%len(cases),item[1]),enumerate(lengths)))
                for index,result in enumerate(refills):
                    check(result,index%len(cases),lengths[index],'concurrent-slot-refill:'+str(index))
            with opened('/v1/completions',{**body(),'stream':True,'stream_options':{'include_usage':True}}) as response:
                chunks=list(events(response, require_terminated=True))
            assert chunks[-1]=='[DONE]'
            parsed=[json.loads(chunk) for chunk in chunks[:-1]]
            assert ''.join(c['text'] for item in parsed for c in item.get('choices',[]))==tokenizer.decode(cases[0]['tokens'],skip_special_tokens=True)
            assert parsed[-1]['usage']['completion_tokens']==len(cases[0]['tokens'])
            report['checks'].append('streaming-content-and-usage')
            with opened('/v1/completions',{**body(),'stream':True,'max_tokens':128}) as response:
                for event in events(response, require_terminated=True):
                    if event=='[DONE]':raise AssertionError('stream ended before cancellation')
                    if any(c.get('text') for c in json.loads(event).get('choices',[])):break
            check(request(1),1,label='cancel-and-reuse-slot')
            for invalid in ({**body(),'temperature':.7},{**body(),'prompt':[151936]},
                            {**body(),'max_tokens':512},{**body(),'model':'missing'},
                            {**body(),'stop':['x']}):
                try:call('/v1/completions',invalid)
                except HTTPError as error:
                    with error:
                        if error.code != 400:
                            raise ValueError('unexpected HTTP rejection status')
                else:raise AssertionError('unsupported request accepted')
            try:call('/v1/models/unload',dict(model='a'))
            except HTTPError as error:
                with error:
                    if error.code != 400:
                        raise ValueError('unexpected HTTP rejection status')
            else:raise AssertionError('distributed unload accepted')
            check(request(),label='invalid-input-and-unsupported-feature-recovery')
            chat=dict(model='a',messages=[dict(role='user',content='What is 2 + 2?')],max_tokens=8,temperature=0)
            first=call('/v1/chat/completions',chat);second=call('/v1/chat/completions',chat)
            assert first['choices']==second['choices'] and first['usage']==second['usage']
            report['checks'].append('resident-chat-repeat')
            before=status('before-idle')
            time.sleep(args.idle_seconds)
            after=status('after-idle')
            if args.idle_seconds>=30:assert after['heartbeats']-before['heartbeats']>=5
            check(request(2),2,label='independent-after-idle')
            assert call('/health')['status']=='ok'
            final=status('final');assert not final['failed'] and not final['closed']
            if final['packed_prefill']:
                assert any(trial['timing']['prefill_batch_size']>1 for trial in report['trials']),'packed prefill never ran'
            report['passed']=True
        except BaseException as error:report['error']=repr(error)
        finally:
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            runtime.close()
    def checked_run(runtime,*positional,**keywords):
        client=threading.Thread(target=qualify,args=(runtime,),daemon=True);client.start()
        try:
            deadline=time.monotonic()+10
            while runtime.requests.qsize()<4 and not runtime.closed.is_set() and time.monotonic()<deadline:
                time.sleep(.001)
            if runtime.requests.qsize()!=4:raise TimeoutError('HTTP proof did not queue its initial four requests')
            return original_run(runtime,*positional,**keywords)
        finally:
            runtime.close();client.join(50)
            if client.is_alive():raise TimeoutError('HTTP validation client did not terminate')
    BoundedHTTPServer.__init__=capture;DistributedRuntime.run=checked_run
    try:serve([args.model,'--quantization',args.quantization,'--alias','a','--port','0','--queue-size','8','--batch-size',str(args.batch_size)])
    finally:BoundedHTTPServer.__init__=original_init;DistributedRuntime.run=original_run
    if not report['passed']:raise AssertionError(report.get('error','HTTP validation failed'))
    print(json.dumps(dict(passed=True,checks=len(report['checks']),idle_seconds=args.idle_seconds)),flush=True)


def main():
    args=parser().parse_args()
    if not 0<=args.idle_seconds<=45:raise ValueError('idle proof must be between 0 and 45 seconds')
    if args.worker:return worker(args)
    from distributed_job import run_job
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node','2',
        '--run-path',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    try:
        result=run_job(command,timeout=240)
        print(result.stdout)
    except subprocess.CalledProcessError as error:
        print(error.stdout,file=sys.stderr);print(error.stderr,file=sys.stderr);raise


if __name__=='__main__':main()
