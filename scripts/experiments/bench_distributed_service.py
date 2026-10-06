"""Matched batch-one/four HTTP comparison on an owned two-rank GPU pair.

Run from the repository root with explicit CUDA UUIDs. Startup/capture is
excluded. The four server launches are counterbalanced; no other GPU test may
run concurrently. Worker mode also supports a bounded cross-host launch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
from urllib.request import urlopen


def worker(args):
    from dllm.distributed_server_cli import main as serve
    options=[args.model,'--quantization',args.quantization,'--alias','a','--port','0',
             '--batch-size',str(args.batch_size)]
    if os.environ['RANK']!='0':serve(options);return
    from dllm.distributed_serving import DistributedRuntime
    from dllm.server import BoundedHTTPServer
    original_init=BoundedHTTPServer.__init__;original_run=DistributedRuntime.run
    base=None
    report=dict(passed=False,batch_size=args.batch_size,quantization=args.quantization,
                model=args.model,context=512,output_tokens=32,results=[],
                devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                         ('dllm/distributed.py','dllm/distributed_serving.py','dllm/distributed_server_cli.py')})
    def capture(server,*positional,**keywords):
        nonlocal base
        original_init(server,*positional,**keywords);base=f'http://127.0.0.1:{server.server_port}'
    def client(runtime):
        try:
            with urlopen(base+'/v1/models/status',timeout=10) as response:report['status']=json.load(response)
            for concurrency in (1,2,4):
                result=subprocess.run([sys.executable,'scripts/compare_llama.py',base+'/v1',
                    '--engine','dllm','--concurrency',str(concurrency),'--max-tokens','32',
                    '--prompt-token-ids',args.prompt],capture_output=True,text=True,check=True,timeout=120)
                report['results'].append(json.loads(result.stdout))
            report['passed']=True
        except BaseException as error:
            report['error']=repr(error)
            if isinstance(error,subprocess.CalledProcessError):report['stderr']=error.stderr
        finally:
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n');runtime.close()
    def measured(runtime,*positional,**keywords):
        thread=threading.Thread(target=client,args=(runtime,),daemon=True);thread.start()
        try:return original_run(runtime,*positional,**keywords)
        finally:runtime.close();thread.join(10)
    BoundedHTTPServer.__init__=capture;DistributedRuntime.run=measured
    try:serve(options)
    finally:BoundedHTTPServer.__init__=original_init;DistributedRuntime.run=original_run
    if not report['passed']:raise AssertionError(report.get('error','HTTP benchmark failed'))


def summarize(trials):
    summary=[]
    for concurrency in (1,2,4):
        for batch in (1,4):
            runs=[run for trial in trials if trial['batch_size']==batch
                  for run in trial['results'] if run['concurrency']==concurrency]
            samples=[sample for run in runs for sample in run['trials'] if not sample['warmup']]
            rounds=[sample for run in runs for sample in run['rounds'] if not sample['warmup']]
            summary.append(dict(batch_size=batch,concurrency=concurrency,samples=len(samples),
                first_text_ms=1000*statistics.median(sample['first_output_seconds'] for sample in samples),
                request_ms=1000*statistics.median(sample['total_seconds'] for sample in samples),
                aggregate_tokens_per_second=statistics.median(sample['aggregate_output_tokens_per_second'] for sample in rounds)))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--batch-size',type=int,choices=(1,4),default=1)
    args=p.parse_args()
    if args.worker:return worker(args)
    from scripts.distributed_job import run_job
    report=dict(passed=False,scope='Same model, precision, GPU pair and HTTP workload; batch capacity only changes.',trials=[])
    output=Path(args.output)
    try:
        reference=None
        for index,batch in enumerate((1,4,4,1)):
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
        report['summary']=summarize(report['trials'])
        report['passed']=True
    finally:output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
