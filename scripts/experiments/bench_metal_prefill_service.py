"""Counterbalanced real HTTP comparison of old and bounded Metal prefill policies."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

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
    p.add_argument('--quantization',choices=('q4_0','q8_0'),required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    report=dict(passed=False,model=args.model,quantization=args.quantization,context=512,
        concurrency=1,output_tokens=32,device='Apple M4',trials=[],
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('dllm/metal_projection_policy.py','dllm/quantized_metal.py','dllm/quantized_metal_q4.py')},
        scope='Same model and precision; expanded versus bounded packed prefill. Accumulation order differs; exact generated text and usage checked.')
    env={**os.environ,'DLLM_DEVICE':'mps','DLLM_QUANTIZATION':args.quantization,
         'DLLM_METAL_WEIGHT_DTYPE':'float16','OMP_NUM_THREADS':'4'}
    ids=json.loads(Path(args.prompt).read_text());assert len(ids)>=16
    texts={}
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-prefill-http-') as directory:
            directory=Path(directory)
            for mode in ('expanded','default','default','expanded'):
                logpath=directory/'server.log'
                with logpath.open('w') as log:
                    process=subprocess.Popen([sys.executable,__file__,'--server-mode',mode,'serve',
                        '--device','mps','--port','0','--context','512','--batch-size','4',
                        '--model','a='+args.model],env=env,stdout=log,stderr=log,start_new_session=True)
                    try:
                        deadline=time.monotonic()+120
                        while time.monotonic()<deadline and process.poll() is None:
                            lines=[line for line in logpath.read_text().splitlines() if line.startswith('dllm listening on ')]
                            if lines:base=lines[-1].split()[-1];break
                            time.sleep(.1)
                        else:raise RuntimeError('server did not become ready')
                        for length in (8,16):
                            promptfile=directory/'prompt.json';promptfile.write_text(json.dumps(ids[:length]))
                            result=subprocess.run([sys.executable,'scripts/compare_llama.py',base+'/v1',
                                '--engine','dllm','--concurrency','1','--max-tokens','32',
                                '--prompt-token-ids',str(promptfile)],env=env,capture_output=True,text=True,check=True,timeout=150)
                            data=json.loads(result.stdout)
                            for trial in data['trials']:
                                assert trial['usage']['prompt_tokens']==length
                                assert trial['usage']['completion_tokens']==32
                                texts.setdefault(length,trial['text']);assert trial['text']==texts[length]
                            report['trials'].append(dict(mode=mode,prompt_tokens=length,result=data))
                            print(mode,length,'complete',flush=True)
                    except BaseException:
                        print(logpath.read_text(),file=sys.stderr);raise
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(20)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(10)
        report['summary']=[]
        for length in (8,16):
            for mode in ('expanded','default'):
                selected=[item['result'] for item in report['trials'] if item['mode']==mode and item['prompt_tokens']==length]
                trials=[r for d in selected for r in d['trials'] if not r['warmup']]
                rounds=[r for d in selected for r in d['rounds'] if not r['warmup']]
                report['summary'].append(dict(mode=mode,prompt_tokens=length,samples=len(trials),
                    first_text_ms=1000*statistics.median(r['first_output_seconds'] for r in trials),
                    request_ms=1000*statistics.median(r['total_seconds'] for r in trials),
                    aggregate_tokens_per_second=statistics.median(r['aggregate_output_tokens_per_second'] for r in rounds)))
        report['passed']=True
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='--server-mode':
        mode=sys.argv[2];sys.argv=[sys.argv[0],*sys.argv[3:]]
        if mode=='expanded':
            from dllm import quantized_metal,quantized_metal_q4
            for module in (quantized_metal,quantized_metal_q4):
                module.use_packed_projection=lambda rows,outputs,width,fp32:rows<=4
        elif mode!='default':raise ValueError('unknown server mode')
        import runpy
        runpy.run_module('dllm',run_name='__main__')
    else:main()
