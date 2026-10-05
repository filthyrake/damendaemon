"""Sequential, same-GGUF HTTP comparison with verified per-slot context."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request,urlopen

from tokenizers import Tokenizer
from dllm.bench import events,llama_context
from dllm.gguf_model import plan


def digest(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    comparison=parser.add_mutually_exclusive_group(required=True)
    comparison.add_argument('--llama-server')
    comparison.add_argument('--residual-library',help='compare production with the isolated residual-int8 candidate')
    parser.add_argument('--native-library',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise RuntimeError('select exactly one GPU UUID before benchmarking')
    model=Path(args.model).resolve();library=Path(args.native_library).resolve()
    llama=Path(args.llama_server).resolve() if args.llama_server else None
    residual=Path(args.residual_library).resolve() if args.residual_library else None
    output=Path(args.output)
    tokenizer=Tokenizer.from_str(plan(model).tokenizer_json)
    base_tokens=tokenizer.encode('Binary search repeatedly halves a sorted range. A detailed explanation follows: ',
                                add_special_tokens=True).ids
    prompts={length:(base_tokens*((length+len(base_tokens)-1)//len(base_tokens)))[:length] for length in (16,129,400)}
    report=dict(model=str(model),model_sha256=digest(model),native_library=str(library),
        native_library_sha256=digest(library),llama_server=str(llama) if llama else None,llama_server_sha256=digest(llama) if llama else None,
        residual_library=str(residual) if residual else None,
        residual_library_sha256=digest(residual) if residual else None,
        experimental_arithmetic='two dyadic int8 activation terms; FP16 single-row width-2048 K-quants' if residual else None,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],context_per_slot=512,slots=4,kv_dtype='float16',
        dllm_precision='mixed: FP16 matrices, FP32 residual/nonlinear',output_tokens=16,
        prompts=prompts,warmup_rounds=3,measured_rounds=6,runs=[])
    environment=os.environ.copy()
    environment.update(DLLM_DEVICE='cuda',DLLM_KQUANT_CUDA_LIBRARY=str(library),DLLM_LLAMA_PRECISION='mixed',
        DLLM_SPECULATIVE_TOKENS='0',DLLM_QUANTIZATION='none',OMP_NUM_THREADS='4')
    production_residual=os.environ.get('DLLM_KQUANT_RESIDUAL_LIBRARY')
    if production_residual:
        report['production_residual_library']=dict(path=production_residual,sha256=digest(production_residual))
    report['experimental_batch2_q6']=bool(residual and os.environ.get('DLLM_EXPERIMENTAL_RESIDUAL_BATCH2')=='1')
    if report['experimental_batch2_q6']:
        report['experimental_arithmetic']+='; batch-two pure Q6 at (width, outputs)=(8192,2048) or (2048,128256)'
    counterpart='residual-int8-v2' if residual else 'llama.cpp'
    if residual:environment['DLLM_EXPERIMENTAL_RESIDUAL_LIBRARY']=str(residual)
    try:
        for phase,engine in enumerate(('dllm',counterpart,counterpart,'dllm')):
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            log_path=output.with_name(output.stem+f'-{phase}-{engine}.log')
            if engine!='llama.cpp':
                module='scripts.experiments.serve_residual' if engine=='residual-int8-v2' else 'dllm'
                command=[sys.executable,'-m',module,'serve','--model',f'a={model}','--port',str(port),
                    '--context','512','--batch-size','4','--kv-cache-tokens','0','--gpu-models','1',
                    '--prefill-chunk-size','0','--prefix-cache-tokens','0']
            else:
                command=[str(llama),'--model',str(model),'--alias','a','--host','127.0.0.1','--port',str(port),
                    '--ctx-size','2048','--parallel','4','--kv-unified-per-slot','512','--gpu-layers','99',
                    '--threads','4','--threads-batch','4','--flash-attn','on','--cache-type-k','f16',
                    '--cache-type-v','f16','--no-webui']
            run=dict(phase=phase,engine=engine,command=command,log=str(log_path),cases=[])
            report['runs'].append(run)
            print(f'Starting phase {phase}: {engine}',file=sys.stderr,flush=True)
            with log_path.open('w') as log:
                process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+180
                    while True:
                        if process.poll() is not None:raise RuntimeError(f'{engine} exited; see {log_path}')
                        try:
                            with urlopen(base+'/health',timeout=2) as response:
                                if response.status==200:break
                        except URLError:pass
                        if time.monotonic()>deadline:raise TimeoutError(f'{engine} startup timed out')
                        time.sleep(.2)
                    if engine=='llama.cpp':
                        run['context']=llama_context(base,512)
                        assert run['context']['total_slots']==4,run['context']
                    def request(ids):
                        body=dict(model='a',prompt=ids,temperature=0,max_tokens=16,stream=True,
                                  stream_options={'include_usage':True})
                        if engine=='llama.cpp':body['cache_prompt']=False
                        started=time.perf_counter();first=None;usage=None;text='';done=False
                        with urlopen(Request(base+'/v1/completions',data=json.dumps(body).encode(),
                                headers={'Content-Type':'application/json'}),timeout=300) as response:
                            for event in events(response):
                                if event=='[DONE]':done=True;break
                                value=json.loads(event)
                                if value.get('error'):raise RuntimeError(value['error'])
                                for choice in value.get('choices',[]):
                                    content=choice.get('text','')
                                    if content:
                                        if first is None:first=time.perf_counter()-started
                                        text+=content
                                if value.get('usage'):usage=value['usage']
                        elapsed=time.perf_counter()-started
                        assert done and first is not None and usage,dict(done=done,usage=usage,text=text)
                        assert usage['prompt_tokens']==len(ids) and usage['completion_tokens']==16,usage
                        return dict(seconds=elapsed,first_output_seconds=first,
                            post_first_output_tokens_per_second=15/(elapsed-first),usage=usage,text=text)
                    for length,concurrency in ((16,1),(129,1),(400,1),(129,2)):
                        print(f'Phase {phase}: input={length}, concurrency={concurrency}',file=sys.stderr,flush=True)
                        case=dict(prompt_tokens=length,concurrency=concurrency,rounds=[])
                        run['cases'].append(case)
                        with ThreadPoolExecutor(concurrency) as pool:
                            for trial in range(9):
                                started=time.perf_counter()
                                values=list(pool.map(request,[prompts[length]]*concurrency))
                                elapsed=time.perf_counter()-started
                                case['rounds'].append(dict(warmup=trial<3,seconds=elapsed,requests=values,
                                    aggregate_output_tokens_per_second=16*concurrency/elapsed))
                        measured=[r for r in case['rounds'] if not r['warmup']]
                        case['median_seconds']=statistics.median(r['seconds'] for r in measured)
                        case['median_aggregate_output_tokens_per_second']=statistics.median(
                            r['aggregate_output_tokens_per_second'] for r in measured)
                finally:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=20)
            output.write_text(json.dumps(report,indent=2)+'\n')
        texts={}
        for run in report['runs']:
            for case in run['cases']:
                key=(case['prompt_tokens'],case['concurrency'])
                for trial in case['rounds']:
                    for request in trial['requests']:
                        texts.setdefault(key,set()).add(request['text'])
        report['all_generated_texts_equal']=all(len(values)==1 for values in texts.values())
        assert report['all_generated_texts_equal'], 'generated text differs between runs'
        report['passed']=True
    except Exception as error:
        report['passed']=False;report['error']=repr(error)
        raise
    finally:output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
