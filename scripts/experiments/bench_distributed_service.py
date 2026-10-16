"""Matched batch-one/four HTTP comparison on an owned two-rank GPU pair.

Run from the repository root with explicit CUDA UUIDs. Startup/capture is
excluded. The four server launches are counterbalanced; no other GPU test may
run concurrently. Worker mode also supports a bounded cross-host launch.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from urllib.request import urlopen


def worker(args):
    from dllm.distributed_server_cli import main as serve
    options=[args.model,'--quantization',args.quantization,'--alias','a','--port','0',
             '--batch-size',str(args.batch_size)]
    if os.environ['RANK']!='0':serve(options);return
    from dllm.distributed_serving import DistributedRuntime
    from dllm.server import BoundedHTTPServer
    original_init=BoundedHTTPServer.__init__;original_run=DistributedRuntime.run
    original_publish=DistributedRuntime._publish
    timings=[]
    def observed_publish(request,event):
        if isinstance(event,dict) and 'timing' in event:timings.append(event['timing'])
        return original_publish(request,event)
    base=None
    report=dict(passed=False,batch_size=args.batch_size,quantization=args.quantization,
                model=args.model,context=512,output_tokens=32,results=[],
                devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                         ('dllm/distributed.py','dllm/distributed_serving.py','dllm/distributed_server_cli.py',
                          'dllm/batch.py','dllm/layer_stage.py','dllm/partition_policy.py','dllm/q4_stage.py','dllm/q4_stage_kernels.py','dllm/kernels.py','dllm/qwen2.py',__file__)})
    def capture(server,*positional,**keywords):
        nonlocal base
        original_init(server,*positional,**keywords);base=f'http://127.0.0.1:{server.server_port}'
    def client(runtime):
        try:
            with urlopen(base+'/v1/models/status',timeout=10) as response:report['status']=json.load(response)
            report['packed_prefill']=report['status']['packed_prefill']
            report['q4_stage_adjacent']=report['status']['weight_layout']=='q4-adjacent-body'
            for concurrency in (1,2,4):
                start=len(timings)
                result=subprocess.run([sys.executable,'scripts/compare_llama.py',base+'/v1',
                    '--engine','dllm','--concurrency',str(concurrency),'--max-tokens','32',
                    '--prompt-token-ids',args.prompt],capture_output=True,text=True,check=True,timeout=120)
                measured=json.loads(result.stdout)
                observed=timings[start:]
                assert len(observed)==6*concurrency,'missing server timing records'
                measured['server_timings']=[dict(warmup=index<concurrency,**value) for index,value in enumerate(observed)]
                report['results'].append(measured)
            report['passed']=True
        except BaseException as error:
            report['error']=repr(error)
            if isinstance(error,subprocess.CalledProcessError):report['stderr']=error.stderr
        finally:
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n');runtime.close()
    def measured(runtime,*positional,**keywords):
        report['runtime_row_bounds']=bool(getattr(runtime.pair.stage.q4_linear,'runtime_rows',False))
        report['runtime_rope_tokens']=runtime.pair.stage.runtime_rope_tokens
        # Arrival timing can make a client's first "warmup" use one slot even
        # when later measured requests form a new packed shape. Warm every
        # possible group on the GPU owner before timing the HTTP rounds.
        prompt=json.loads(Path(args.prompt).read_text())
        started=time.perf_counter()
        for size in range(1,args.batch_size+1):
            slots=list(range(size))
            if size>1 and runtime.pair.packed_prefill:
                runtime.pair.prefill_batch([prompt]*size,slots)
            else:
                for slot in slots:runtime.pair.prefill(prompt,slot)
            runtime.pair.release(slots)
        report['shape_warmup']=dict(group_sizes=list(range(1,args.batch_size+1)),
            prompt_tokens=len(prompt),seconds=time.perf_counter()-started,
            scope='Policy-reachable prefill shapes warmed on the GPU owner before HTTP measurement')
        thread=threading.Thread(target=client,args=(runtime,),daemon=True);thread.start()
        try:return original_run(runtime,*positional,**keywords)
        finally:runtime.close();thread.join(10)
    BoundedHTTPServer.__init__=capture;DistributedRuntime.run=measured
    DistributedRuntime._publish=staticmethod(observed_publish)
    try:serve(options)
    finally:
        BoundedHTTPServer.__init__=original_init;DistributedRuntime.run=original_run
        DistributedRuntime._publish=staticmethod(original_publish)
    if not report['passed']:raise AssertionError(report.get('error','HTTP benchmark failed'))


def summarize(trials,compare='capacity'):
    key={'capacity':'batch_size','q4-policy':'q4_stage_adjacent',
         'prefill':'packed_prefill','runtime-rows':'runtime_row_bounds','runtime-rope':'runtime_rope_tokens'}[compare]
    modes=(1,4) if compare=='capacity' else (False,True)
    summary=[]
    for concurrency in (1,2,4):
        for mode in modes:
            runs=[run for trial in trials if trial[key]==mode
                  for run in trial['results'] if run['concurrency']==concurrency]
            samples=[sample for run in runs for sample in run['trials'] if not sample['warmup']]
            rounds=[sample for run in runs for sample in run['rounds'] if not sample['warmup']]
            summary.append(dict(**{key:mode},concurrency=concurrency,samples=len(samples),
                first_text_ms=1000*statistics.median(sample['first_output_seconds'] for sample in samples),
                request_ms=1000*statistics.median(sample['total_seconds'] for sample in samples),
                aggregate_tokens_per_second=statistics.median(sample['aggregate_output_tokens_per_second'] for sample in rounds)))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--batch-size',type=int,choices=(1,4),default=1)
    p.add_argument('--compare',choices=('capacity','prefill','q4-policy','runtime-rows','runtime-rope'),default='capacity')
    args=p.parse_args()
    if args.compare in ('q4-policy','runtime-rows','runtime-rope') and args.quantization!='q4_0':raise ValueError('Q4 policy comparison requires Q4 weights')
    if args.worker:return worker(args)
    from scripts.distributed_job import run_job
    report=dict(passed=False,compare=args.compare,scope='Same model, precision, GPU pair and HTTP workload; '+
                ('batch capacity' if args.compare=='capacity' else 'both Q4 body layout and stage prefill packing'
                 if args.compare=='q4-policy' else 'Q4 runtime row bounds' if args.compare=='runtime-rows'
                 else 'Q4 runtime rotary token stride' if args.compare=='runtime-rope'
                 else 'stage prefill packing')+' changes.',trials=[])
    output=Path(args.output)
    previous={key:os.environ.get(key) for key in ('DLLM_PARTITION_PACKED_PREFILL','DLLM_Q4_STAGE_ADJACENT','DLLM_Q4_STAGE_RUNTIME_ROWS','DLLM_Q4_STAGE_RUNTIME_ROPE')}
    try:
        reference=None
        modes=(1,4,4,1) if args.compare=='capacity' else (0,1,1,0)
        for index,mode in enumerate(modes):
            batch=mode if args.compare=='capacity' else 4
            if args.compare=='runtime-rope':
                os.environ.update(DLLM_PARTITION_PACKED_PREFILL='1',DLLM_Q4_STAGE_ADJACENT='1',DLLM_Q4_STAGE_RUNTIME_ROWS='1',DLLM_Q4_STAGE_RUNTIME_ROPE=str(mode))
            elif args.compare=='runtime-rows':
                os.environ.update(DLLM_PARTITION_PACKED_PREFILL='1',DLLM_Q4_STAGE_ADJACENT='1',DLLM_Q4_STAGE_RUNTIME_ROWS=str(mode),DLLM_Q4_STAGE_RUNTIME_ROPE='0')
            elif args.compare!='capacity':os.environ['DLLM_PARTITION_PACKED_PREFILL']=str(mode)
            if args.compare=='q4-policy':os.environ['DLLM_Q4_STAGE_ADJACENT']=str(mode)
            trial=output.with_name(output.stem+f'-{index}.json')
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node','2',
                '--run-path',str(Path(__file__).resolve()),args.model,args.prompt,'--quantization',args.quantization,
                '--output',str(trial),'--batch-size',str(batch),'--worker']
            result=run_job(command,timeout=240);print(result.stdout,flush=True)
            data=json.loads(trial.read_text());assert data['passed']
            for run in data['results']:
                for sample in run['trials']:
                    content=(sample['text'],sample['usage'])
                    if reference is None:reference=content
                    assert reference==content,'generated text or usage differs'
            report['trials'].append(data)
        report['summary']=summarize(report['trials'],args.compare)
        report['passed']=True
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
