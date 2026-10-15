"""Check continuation output against an independent token trace through normal HTTP.

Engine text parity is recorded separately; HF token agreement does not imply
that another engine using different arithmetic generates the same words.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from dllm.gguf_sentencepiece import restore_tokenizer
from dllm.bench import events
from dllm.gguf_model import plan


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--benchmark',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    benchmark=json.loads(Path(args.benchmark).read_text());reference=json.loads(Path(args.reference).read_text())
    if not reference['passed'] or reference['sha256']!=benchmark['model_sha256']:
        raise ValueError('requires passing same-checkpoint independent token trace')
    with Path(args.model).open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    if checksum!=reference['sha256']:raise ValueError('model differs from reference')
    source=plan(args.model);tokenizer=restore_tokenizer(source.tokenizer_json, source.sentencepiece_prefix)
    decode=getattr(tokenizer,'decode_generated',tokenizer.decode)
    output=Path(args.output);report=dict(passed=False,model_sha256=checksum,cases=[],
        reference=args.reference,benchmark=args.benchmark,scope='normal HTTP continuation decoding against independently qualified token traces')
    with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    command=[sys.executable,'-m','dllm','serve','--model',f'a={Path(args.model).resolve()}',
             '--port',str(port),'--context','512','--batch-size','4','--kv-cache-tokens','0',
             '--prefill-chunk-size','0','--prefix-cache-tokens','0']
    report['command']=command
    with output.with_suffix('.server.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None:raise RuntimeError('server exited')
                try:
                    with urlopen(base+'/health',timeout=2) as response:
                        if response.status==200:break
                except URLError:pass
                if time.monotonic()>deadline:raise TimeoutError('server startup')
                time.sleep(.2)
            for index,key in enumerate(sorted(benchmark['prompts'],key=int)):
                ids=benchmark['prompts'][key]
                trace=next(c for c in reference['cases'] if not c['graph'] and c['case']==index)
                if trace['prompt_tokens']!=len(ids) or trace['greedy_mismatches']:raise ValueError('reference case mismatch')
                expected=decode(trace['reference_tokens'],skip_special_tokens=True)
                body=dict(model='a',prompt=ids,temperature=0,max_tokens=benchmark['output_tokens'])
                case=dict(prompt_tokens=len(ids),expected=expected);report['cases'].append(case)
                for streaming in (False,True):
                    request=Request(base+'/v1/completions',data=json.dumps({**body,'stream':streaming}).encode(),
                                    headers={'Content-Type':'application/json'})
                    with urlopen(request,timeout=300) as response:
                        if streaming:
                            chunks=list(events(response, require_terminated=True));assert chunks[-1]=='[DONE]'
                            text=''.join(c.get('text','') for item in chunks[:-1] for c in json.loads(item)['choices'])
                        else:
                            value=json.load(response);text=value['choices'][0]['text'];case['timing']=value['dllm_timing']
                    case['stream' if streaming else 'nonstream']=dict(text=text,equal=text==expected)
                    assert text==expected,case
                native=next(c for run in benchmark['runs'] if run['engine']=='llama.cpp'
                            for c in run['cases'] if c['prompt_tokens']==len(ids) and c['concurrency']==1)
                case['llama_reference_equal']=expected==native['rounds'][3]['requests'][0]['text']
            report['passed']=True
        except Exception as error:
            report['error']=repr(error);raise
        finally:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))))


if __name__=='__main__':main()
