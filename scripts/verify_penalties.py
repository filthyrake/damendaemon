"""Independent scalar penalty oracle plus live CUDA/MPS completion checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import torch
from dllm.sampling import Sampling,Sampler
from dllm.device import empty_cache
from dllm.runtime import require_gpu
from dllm.bench import events
from tokenizers import Tokenizer


def oracle(logits,prompt,output,options):
    values=logits.float().cpu().tolist()
    seen=set(prompt+output)
    counts=Counter(output)
    for token,value in enumerate(values):
        if token in seen:
            value=value/options.repetition_penalty if value>0 else value*options.repetition_penalty
        value-=counts[token]*options.frequency_penalty+(options.presence_penalty if counts[token] else 0)
        values[token]=value+(options.logit_bias or {}).get(token,0)
    return torch.tensor(values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--device',choices=('cuda','mps'),default='cuda')
    args=p.parse_args()
    os.environ['DLLM_DEVICE']=args.device
    require_gpu()
    torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(18)
    cases=0
    for repetition in (.5,1,1.8):
        for presence,frequency in ((0,0),(.7,.4),(-1,-.3)):
            options=Sampling(presence_penalty=presence,frequency_penalty=frequency,
                             repetition_penalty=repetition,logit_bias={'7':-12,'11':9})
            sampler=Sampler(options)
            prompt=[2,3,3,7];output=[3,3,11,8]
            sampler.prepare(prompt,257)
            for token in output:sampler.observe(token)
            logits=torch.randn(257,generator=generator).to(args.device)
            before=logits.clone()
            expected=oracle(logits,prompt,output,options)
            actual=sampler.adjust(logits).cpu()
            assert torch.allclose(actual,expected,atol=2e-6,rtol=1e-6)
            assert torch.equal(before,logits)
            assert sampler.select(logits).item()==expected.argmax().item()
            cases+=1
    del sampler,logits,before,actual
    gc.collect();empty_cache()
    tokenizer=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'))
    config=json.loads((Path(args.model)/'config.json').read_text())
    target=tokenizer.encode(' dog',add_special_tokens=False).ids
    assert len(target)==1
    forced={'logit_bias':{str(target[0]):100},'presence_penalty':1,'frequency_penalty':.5,'repetition_penalty':1.1}
    history_options={'presence_penalty':.8,'frequency_penalty':.6,'repetition_penalty':1.3}
    def history_reference():
        from dllm.runtime import Runtime
        from dllm.prompts import encode
        runtime=Runtime({'a':args.model},context=512)
        with runtime.session('a') as (executor,_,__):
            prompt=encode(executor.model,None,'The animal is',None)
            output=[]
            logits=executor.prefill(prompt).squeeze(0)
            for step in range(8):
                token=oracle(logits,prompt,output,Sampling(**history_options)).argmax().item()
                output.append(token)
                if token in executor.model.eos:break
                if step<7:logits=executor.decode(token)[0].squeeze(0)
        actual=list(runtime.generate('a',None,max_tokens=8,raw_prompt='The animal is',**history_options))
        assert [value['token'] for value in actual if 'token' in value]==output
        return tokenizer.decode(output,skip_special_tokens=True),output
    history_text,history_tokens=history_reference()
    gc.collect();empty_cache()
    with tempfile.TemporaryDirectory(prefix='dllm-penalties-') as directory:
        logpath=Path(directory)/'server.log'
        with logpath.open('w') as log:
            process=subprocess.Popen([sys.executable,'-m','dllm','serve','--device',args.device,
                '--port','0','--context','512','--batch-size','4','--model','a='+args.model],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+120
                while time.monotonic()<deadline and process.poll() is None:
                    lines=[v for v in logpath.read_text().splitlines() if v.startswith('dllm listening on ')]
                    if lines:base=lines[-1].split()[-1];break
                    time.sleep(.1)
                else:raise RuntimeError(logpath.read_text())
                def request(raw=False,stream=False,**options):
                    body={'model':'a','max_tokens':8,'stream':stream,**options}
                    body.update({'prompt':'The animal is'} if raw else {'messages':[{'role':'user','content':'Name an animal.'}]})
                    route='/v1/completions' if raw else '/v1/chat/completions'
                    with urlopen(Request(base+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
                        if stream:
                            chunks=list(events(response, require_terminated=True));assert chunks[-1]=='[DONE]'
                            choices=[json.loads(v)['choices'][0] for v in chunks[:-1]]
                            return ''.join(c.get('text','') if raw else c.get('delta',{}).get('content','') for c in choices)
                        value=json.load(response)
                        return value['choices'][0].get('text') if raw else value['choices'][0]['message']['content']
                expected=tokenizer.decode(target*8,skip_special_tokens=True)
                for raw in (False,True):
                    assert request(raw=raw,**forced)==expected
                    assert request(raw=raw,stream=True,**forced)==expected
                assert request(raw=True,**history_options)==history_text
                assert request(raw=True,stream=True,**history_options)==history_text
                baseline=request()
                with ThreadPoolExecutor(4) as pool:
                    futures=[pool.submit(request,**(forced if i%2 else {})) for i in range(4)]
                    assert [f.result() for f in futures]==[baseline,expected,baseline,expected]
                for invalid in ({'logit_bias':{str(config['vocab_size']):1}}, {'presence_penalty':3}, {'repetition_penalty':0}):
                    try:request(**invalid);raise AssertionError('invalid control accepted')
                    except HTTPError as error:
                        with error:
                            if error.code != 400:
                                raise ValueError('unexpected HTTP rejection status')
                assert request()==baseline
                print(json.dumps({'passed':True,'device':args.device,'model':args.model,
                    'independent_numeric_cases':cases,'chat_raw_streaming':True,
                    'mixed_request_isolation':True,'invalid_request_recovery':True,
                    'history_oracle_tokens':history_tokens,'history_oracle_text':history_text,
                    'forced_token_id':target[0],'forced_text':expected},indent=2))
            except Exception:
                print(logpath.read_text(),file=sys.stderr);raise
            finally:
                process.terminate()
                try:process.wait(30)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
