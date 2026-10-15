"""Live prefix hit/miss, model-switch isolation and warmed HTTP latency proof."""
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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    p.add_argument('--chunk-size',type=int,default=0)
    p.add_argument('--host-gib',type=float,default=8)
    args=p.parse_args()
    tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'))
    tokenizer.no_padding();tokenizer.no_truncation()
    prompt=tokenizer.encode('A blue bird flies over a green field. '*200,add_special_tokens=True).ids[:1536]
    assert len(prompt)==1536
    rows=[];baseline=None
    for budget in (0,4096):
        with tempfile.TemporaryDirectory(prefix='dllm-prefix-service-') as directory:
            logpath=Path(directory)/'server.log'
            with logpath.open('w') as log:
                process=subprocess.Popen([sys.executable,'-m','dllm','serve','--port','0','--device',args.device,
                    '--context','2048','--batch-size','4','--gpu-models','1','--host-models','2',
                    '--host-gib',str(args.host_gib),
                    '--prefix-cache-tokens',str(budget),'--prefill-chunk-size',str(args.chunk_size),'--model','a='+args.model,'--model','b='+args.model],stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+120
                    while time.monotonic()<deadline and process.poll() is None:
                        lines=[v for v in logpath.read_text().splitlines() if v.startswith('dllm listening on ')]
                        if lines:base=lines[-1].split()[-1];break
                        time.sleep(.1)
                    else:raise RuntimeError(logpath.read_text())
                    def request(alias='a',tokens=prompt,**options):
                        body={'model':alias,'prompt':tokens,'max_tokens':8,'temperature':0,**options}
                        start=time.perf_counter()
                        with urlopen(Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:result=json.load(response)
                        result['measured_http_seconds']=time.perf_counter()-start
                        return result
                    cold=request()
                    assert cold['dllm_timing']['cached_prompt_tokens']==0
                    baseline=cold['choices'] if baseline is None else baseline
                    assert cold['choices']==baseline
                    trials=[request() for _ in range(5)]
                    for result in trials:
                        assert result['choices']==baseline
                        assert result['usage']['prompt_tokens']==1536
                        assert result['dllm_timing']['cached_prompt_tokens']==(1536 if budget else 0)
                    with ThreadPoolExecutor(4) as pool:concurrent=list(pool.map(lambda _:request(),range(4)))
                    assert all(r['choices']==baseline for r in concurrent)
                    # Close a real streaming client after output, then reuse its prefix.
                    body={'model':'a','prompt':prompt,'max_tokens':64,'temperature':0,'stream':True}
                    saw_output=False
                    with urlopen(Request(base+'/v1/completions',data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'}),timeout=180) as response:
                        for event in events(response, require_terminated=True):
                            if event=='[DONE]':break
                            chunk=json.loads(event)
                            if chunk.get('error'):raise RuntimeError(chunk['error'])
                            if any(choice.get('text') for choice in chunk.get('choices',[])):
                                saw_output=True
                                break
                    assert saw_output
                    assert request()['choices']==baseline
                    # The second alias uses identical weights but owns independent cache state.
                    other=request('b')
                    restored=request()
                    assert other['dllm_timing']['cached_prompt_tokens']==0
                    assert restored['dllm_timing']['cached_prompt_tokens']==0
                    assert restored['dllm_timing']['source_tier']=='host'
                    assert restored['choices']==baseline
                    divergent=request(tokens=prompt+[1,2,3])
                    assert divergent['dllm_timing']['cached_prompt_tokens']==(1536 if budget else 0)
                    assert divergent['usage']['prompt_tokens']==1539
                    partial_trials=[request(tokens=prompt+[token,2,3]) for token in range(20,25)]
                    assert all(r['dllm_timing']['cached_prompt_tokens']==(1536 if budget else 0) for r in partial_trials)
                    rows.append({'token_budget':budget,'trials':[{'http_seconds':r['measured_http_seconds'],
                        'timing':r['dllm_timing'],'usage':r['usage']} for r in trials],
                        'median_first_token_ms':statistics.median(r['dllm_timing']['first_token_seconds'] for r in trials)*1000,
                        'median_http_ms':statistics.median(r['measured_http_seconds'] for r in trials)*1000,
                        'partial_trials':[{'timing':r['dllm_timing'],'http_ms':r['measured_http_seconds']*1000,'choices':r['choices']} for r in partial_trials],
                        'median_partial_first_token_ms':statistics.median(r['dllm_timing']['first_token_seconds'] for r in partial_trials)*1000,
                        'divergent_timing':divergent['dllm_timing'],'divergent_http_ms':divergent['measured_http_seconds']*1000,
                        'divergent_choices':divergent['choices'],'model_switch_cache_isolation':True,'concurrent_output_equality':True,'disconnect_recovery':True})
                except Exception:
                    print(logpath.read_text(),file=sys.stderr);raise
                finally:
                    process.terminate()
                    try:process.wait(30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
    assert rows[0]['divergent_choices']==rows[1]['divergent_choices']
    assert [r['choices'] for r in rows[0]['partial_trials']]==[r['choices'] for r in rows[1]['partial_trials']]
    print(json.dumps({'passed':True,'model':args.model,'device':args.device,
        'runtime_environment':{k:v for k,v in os.environ.items() if k.startswith('DLLM_')},
        'prefill_chunk_size':args.chunk_size,'host_gib':args.host_gib,'prompt_tokens':len(prompt),'max_output_tokens':8,'results':rows},indent=2))


if __name__=='__main__':main()
