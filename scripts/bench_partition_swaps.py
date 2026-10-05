"""Matched HTTP host-restore benchmark with distinct independently checked weights."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from urllib.request import Request,urlopen
import torch
from tokenizers import Tokenizer
from scripts.verify_partition_service import cases_from_trace


def main():
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('changed');p.add_argument('--output',required=True)
    p.add_argument('--swaps',type=int,default=6);args=p.parse_args()
    if args.swaps<2 or args.swaps%2:p.error('swaps must be positive and even, at least two')
    traces={alias:torch.load(path,map_location='cpu',weights_only=True) for alias,path in (('a',args.original),('b',args.changed))}
    quantization=traces['a'].get('reference_quantization','none')
    assert traces['b'].get('reference_quantization','none')==quantization
    cases={alias:cases_from_trace(t)[0] for alias,t in traces.items()}
    for alias,case in cases.items():
        eos=json.loads((Path(traces[alias]['path'])/'config.json').read_text())['eos_token_id']
        eos=[eos] if isinstance(eos,int) else eos
        for index,token in enumerate(case['tokens']):
            if token in eos:case['tokens']=case['tokens'][:index+1];break
    tokenizer=Tokenizer.from_file(str(Path(traces['a']['path'])/'tokenizer.json'))
    report=dict(passed=False,quantization=quantization,models={a:t['path'] for a,t in traces.items()},
        devices=os.environ['CUDA_VISIBLE_DEVICES'],context=512,batch_size=4,groups=[],
        scope='same-host two-GPU HTTP host-restore ABBA; distinct-weight correctness fixtures; no cold-load or cross-network claim')
    try:
        for rebind in (False,True,True,False):
            group=dict(rebind=rebind,warmups=[],runs=[]);report['groups'].append(group)
            with tempfile.TemporaryDirectory(prefix='dllm-swap-bench-') as directory:
                logpath=Path(directory)/'server.log'
                with logpath.open('w') as log:
                    env={**os.environ,'DLLM_PARTITION_REBIND':str(int(rebind)),'DLLM_QUANTIZATION':quantization}
                    process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--layer-split','12',
                        '--context','512','--batch-size','4','--gpu-models','1','--host-models','2','--host-gib','8',
                        '--model','a='+traces['a']['path'],'--model','b='+traces['b']['path']],env=env,stdout=log,stderr=log)
                    try:
                        deadline=time.monotonic()+120
                        while time.monotonic()<deadline and process.poll() is None:
                            ready=[s for s in logpath.read_text().splitlines() if s.startswith('dllm listening on ')]
                            if ready:base=ready[-1].split()[-1];break
                            time.sleep(.1)
                        else:raise RuntimeError(logpath.read_text())
                        def call(path,body):
                            with urlopen(Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as response:
                                return json.load(response)
                        for index in range(4+args.swaps):
                            alias='a' if index%2==0 else 'b';case=cases[alias]
                            started=time.perf_counter();loaded=call('/v1/models/load',dict(model=alias));wall=time.perf_counter()-started
                            if index>=4:assert loaded['source_tier']=='host'
                            entry=next(m for m in loaded['models'] if m['id']==alias)
                            assert entry['partition_rebind_count']==(index if rebind else 0)
                            result=call('/v1/completions',dict(model=alias,prompt=case['prompt'],max_tokens=len(case['tokens']),temperature=0))
                            assert result['choices'][0]['text']==tokenizer.decode(case['tokens'],skip_special_tokens=True)
                            assert result['usage']['completion_tokens']==len(case['tokens'])
                            row=dict(alias=alias,activation_seconds=loaded['activation_seconds'],http_seconds=wall,
                                source_tier=loaded['source_tier'],rebind_count=entry['partition_rebind_count'],reference_match=True,
                                prompt_tokens=len(case['prompt']),output_tokens=len(case['tokens']),
                                cuda_allocated_bytes=[d['allocated_bytes'] for d in loaded['cuda_devices']])
                            group['runs' if index>=4 else 'warmups'].append(row)
                        for alias in traces:call('/v1/models/unload',dict(model=alias,release_host=True))
                    except BaseException:
                        print(logpath.read_text(),file=sys.stderr);raise
                    finally:
                        process.terminate()
                        try:process.wait(30)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
        report['summaries']=[]
        for rebind in (False,True):
            rows=[r for g in report['groups'] if g['rebind']==rebind for r in g['runs']]
            report['summaries'].append(dict(rebind=rebind,measured_swaps=len(rows),
                median_activation_seconds=statistics.median(r['activation_seconds'] for r in rows),
                median_http_seconds=statistics.median(r['http_seconds'] for r in rows)))
        report['passed']=True
    finally:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summaries']))


if __name__=='__main__':main()
