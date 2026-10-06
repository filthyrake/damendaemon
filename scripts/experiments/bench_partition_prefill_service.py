"""Counterbalanced sequential/packed stage prefill through real HTTP.

Select exactly two GPU UUIDs and run GPU jobs sequentially. The idle coalescing
window is identical in both modes (zero by default). Startup/capture is excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),required=True)
    p.add_argument('--batch-wait-ms',type=float,default=0)
    p.add_argument('--compare',choices=('prefill','q4-body'),default='prefill')
    p.add_argument('--body-kernel',choices=('single-dot','paired-load','adjacent-gate','adjacent-all'),default='single-dot')
    p.add_argument('--output',required=True);args=p.parse_args()
    if not 0<=args.batch_wait_ms<=10:raise ValueError('invalid coalescing delay')
    if args.compare=='q4-body' and args.quantization!='q4_0':raise ValueError('Q4 body comparison requires Q4 weights')
    modes=('sequential','packed') if args.compare=='prefill' else ('paired','contiguous')
    report=dict(passed=False,model=args.model,quantization=args.quantization,context=512,batch_size=4,
        batch_wait_ms=args.batch_wait_ms,compare=args.compare,output_tokens=32,trials=[],
        devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('dllm/batch.py','dllm/layer_stage.py','dllm/partition.py','dllm/partition_policy.py')},
        scope='Same model, precision, GPU pair and HTTP workload; '+
              ('sequential versus packed stage prefill only.' if args.compare=='prefill' else
               'paired versus contiguous packed Q4 body arithmetic; prompt packing disabled in both.'))
    if args.compare=='q4-body':
        report['body_kernel']=args.body_kernel
        for name in ('dllm/quantized_kernels_q4.py','scripts/experiments/q4_prefill.py',
                     'scripts/experiments/q4_body_single_dot.py','scripts/experiments/q4_adjacent_body.py',
                     'scripts/experiments/run_q4_body.py'):
            report['sources'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    env={**os.environ,'DLLM_DEVICE':'cuda','DLLM_QUANTIZATION':args.quantization,'OMP_NUM_THREADS':'4'}
    reference=None
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-partition-prefill-') as directory:
            logpath=Path(directory)/'server.log'
            for mode in (modes[0],modes[1],modes[1],modes[0]):
                with logpath.open('w') as log:
                    module='scripts.experiments.run_q4_body' if mode=='contiguous' else 'dllm'
                    extra=['--body-kernel',args.body_kernel] if mode=='contiguous' else []
                    process=subprocess.Popen([sys.executable,'-m',module,*extra,'serve','--port','0','--context','512',
                        '--layer-split','12','--batch-size','4','--batch-wait-ms',str(args.batch_wait_ms),
                        '--model','a='+args.model],
                        env={**env,'DLLM_PARTITION_PACKED_PREFILL':str(int(mode=='packed'))},
                        stdout=log,stderr=log,start_new_session=True)
                    try:
                        deadline=time.monotonic()+120
                        while time.monotonic()<deadline and process.poll() is None:
                            lines=[line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                            if lines:base=lines[-1].split()[-1];break
                            time.sleep(.1)
                        else:raise RuntimeError('partition server did not become ready')
                        results=[]
                        for concurrency in (1,2,4):
                            result=subprocess.run([sys.executable,'scripts/compare_llama.py',base+'/v1',
                                '--engine','dllm','--concurrency',str(concurrency),'--max-tokens','32',
                                '--prompt-token-ids',args.prompt],env=env,capture_output=True,text=True,check=True,timeout=150)
                            data=json.loads(result.stdout)
                            for trial in data['trials']:
                                value=(trial['text'],trial['usage'])
                                if reference is None:reference=value
                                assert reference==value,'text or usage changed'
                            results.append(data)
                        report['trials'].append(dict(mode=mode,results=results))
                        print(mode,'complete',flush=True)
                    except BaseException:
                        print(logpath.read_text(),file=sys.stderr);raise
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(20)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(10)
        report['summary']=[]
        for concurrency in (1,2,4):
            for mode in modes:
                runs=[run for trial in report['trials'] if trial['mode']==mode
                      for run in trial['results'] if run['concurrency']==concurrency]
                samples=[sample for run in runs for sample in run['trials'] if not sample['warmup']]
                rounds=[sample for run in runs for sample in run['rounds'] if not sample['warmup']]
                report['summary'].append(dict(mode=mode,concurrency=concurrency,samples=len(samples),
                    first_text_ms=1000*statistics.median(sample['first_output_seconds'] for sample in samples),
                    request_ms=1000*statistics.median(sample['total_seconds'] for sample in samples),
                    aggregate_tokens_per_second=statistics.median(sample['aggregate_output_tokens_per_second'] for sample in rounds)))
        report['passed']=True
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
