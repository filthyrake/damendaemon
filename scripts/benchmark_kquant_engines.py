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
    comparison.add_argument('--q4-residual-library',help='compare native Q4 decode with the isolated integer-dot candidate')
    comparison.add_argument('--residual-library',help='compare production with the isolated residual-int8 candidate')
    comparison.add_argument('--residual-f32-library',help='compare ordinary packed FP32 with the isolated FP32 candidate')
    parser.add_argument('--production-prefill-f32',action='store_true',help='exercise normal per-executor compensated prefill selection')
    parser.add_argument('--prefill-f32',action='store_true',help='compare production FP32 residual decode with experimental compensated prefill')
    parser.add_argument('--production-residual-f32',action='store_true',help='exercise normal CLI per-executor FP32 selection')
    parser.add_argument('--lifecycle',action='store_true',help='also check dllm cancellation, swapping and reloads after timing')
    parser.add_argument('--swap-model', help='optional different GGUF checkpoint for lifecycle swapping')
    parser.add_argument('--native-format', choices=('kquant','q4'), default='kquant', help='native decode library format')
    parser.add_argument('--native-library',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.swap_model and not args.lifecycle:parser.error('--swap-model requires --lifecycle')
    if args.native_format=='q4' and not (args.llama_server or args.q4_residual_library):
        parser.error('--native-format q4 requires --llama-server or --q4-residual-library')
    if args.q4_residual_library and args.native_format!='q4':parser.error('Q4 candidate requires --native-format q4')
    if args.production_prefill_f32:args.prefill_f32=True
    if args.prefill_f32:args.production_residual_f32=True
    if args.production_residual_f32 and not args.residual_f32_library:
        parser.error('--production-residual-f32 requires --residual-f32-library')
    precision='float32' if args.residual_f32_library else 'mixed'
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise RuntimeError('select exactly one GPU UUID before benchmarking')
    model=Path(args.model).resolve();library=Path(args.native_library).resolve()
    llama=Path(args.llama_server).resolve() if args.llama_server else None
    residual_argument=args.residual_library or args.residual_f32_library or args.q4_residual_library
    residual=Path(residual_argument).resolve() if residual_argument else None
    swap_model=Path(args.swap_model).resolve() if args.swap_model else model
    output=Path(args.output)
    tokenizer=Tokenizer.from_str(plan(model).tokenizer_json)
    base_tokens=tokenizer.encode('Binary search repeatedly halves a sorted range. A detailed explanation follows: ',
                                add_special_tokens=True).ids
    prompts={length:(base_tokens*((length+len(base_tokens)-1)//len(base_tokens)))[:length] for length in (16,129,400)}
    report=dict(native_format=args.native_format, model=str(model),model_sha256=digest(model),native_library=str(library),
        native_library_sha256=digest(library),llama_server=str(llama) if llama else None,llama_server_sha256=digest(llama) if llama else None,
        residual_library=str(residual) if residual else None,
        residual_library_sha256=digest(residual) if residual else None,
        experimental_arithmetic=('two dyadic int8 activation terms; FP32 rows 1-4, widths 2048/8192' if precision=='float32'
                                else 'two dyadic int8 activation terms; FP16 single-row width-2048 K-quants') if residual else None,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],context_per_slot=512,slots=4,kv_dtype='float32' if precision=='float32' else 'float16',
        dllm_precision='FP32 matrices, residual/nonlinear and KV' if precision=='float32' else 'mixed: FP16 matrices, FP32 residual/nonlinear',output_tokens=16,
        prompts=prompts,warmup_rounds=3,measured_rounds=6,runs=[])
    if args.q4_residual_library:
        report['experimental_arithmetic']='Q4 single-row integer dot; two dyadic int8 activation terms; FP32 coefficient scaling rather than rounded FP16 coefficients'
    report['swap_model']=dict(path=str(swap_model),sha256=digest(swap_model)) if args.swap_model else None
    report['production_residual_f32']=args.production_residual_f32
    report['experimental_prefill_f32']=args.prefill_f32 and not args.production_prefill_f32
    report['production_prefill_f32']=args.production_prefill_f32
    if args.prefill_f32:
        report['experimental_arithmetic']='FP32 prefill rows >=128; scaled high/low FP16 operands, three products; same production residual decode in both engines'
        report['prefill_source_sha256']=digest(Path(__file__).parent/'experiments/kquant_prefill_f32.py')
        report['prefill_kernel_sha256']=digest(Path(__file__).resolve().parents[1]/'dllm/kquant_prefill_kernels.py')
    environment=os.environ.copy()
    environment.pop('DLLM_KQUANT_RESIDUAL_F32_LIBRARY',None)
    environment.pop('DLLM_KQUANT_PREFILL_F32',None)
    environment.pop('DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY',None)
    environment.pop('DLLM_Q4_CUDA_LIBRARY',None)
    environment.pop('DLLM_KQUANT_CUDA_LIBRARY',None)
    environment['DLLM_Q4_CUDA_LIBRARY' if args.native_format=='q4' else 'DLLM_KQUANT_CUDA_LIBRARY']=str(library)
    environment.update(DLLM_DEVICE='cuda',DLLM_LLAMA_PRECISION=precision,
        DLLM_QWEN3_PRECISION='float32' if precision=='float32' else 'standard',
        DLLM_SPECULATIVE_TOKENS='0',DLLM_QUANTIZATION='none',OMP_NUM_THREADS='4')
    production_residual=os.environ.get('DLLM_KQUANT_RESIDUAL_LIBRARY')
    if production_residual:
        report['production_residual_library']=dict(path=production_residual,sha256=digest(production_residual))
    report['experimental_batch2_q6']=bool(args.residual_library and os.environ.get('DLLM_EXPERIMENTAL_RESIDUAL_BATCH2')=='1')
    if report['experimental_batch2_q6']:
        report['experimental_arithmetic']+='; batch-two pure Q6 at (width, outputs)=(8192,2048) or (2048,128256)'
    counterpart='residual-int8-fp32' if precision=='float32' else 'residual-int8-v2' if residual else 'llama.cpp'
    if args.q4_residual_library:counterpart='q4-residual-int8'
    if residual:
        variable='DLLM_EXPERIMENTAL_RESIDUAL_F32_LIBRARY' if precision=='float32' else 'DLLM_EXPERIMENTAL_RESIDUAL_LIBRARY'
        environment[variable]=str(residual)
    if precision=='float32':environment['NVIDIA_TF32_OVERRIDE']='0'
    if args.prefill_f32:
        counterpart='compensated-prefill-fp32'
        environment['DLLM_KQUANT_RESIDUAL_F32_LIBRARY']=str(residual)
    try:
        for phase,engine in enumerate(('dllm',counterpart,counterpart,'dllm')):
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            log_path=output.with_name(output.stem+f'-{phase}-{engine}.log')
            phase_environment=environment.copy()
            if args.q4_residual_library and engine==counterpart:
                phase_environment['DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY']=str(residual)
            if args.production_residual_f32 and engine==counterpart:
                phase_environment['DLLM_KQUANT_RESIDUAL_F32_LIBRARY']=str(residual)
            if args.production_prefill_f32 and engine==counterpart:
                phase_environment['DLLM_KQUANT_PREFILL_F32']='1'
            if engine!='llama.cpp':
                module={'q4-residual-int8':'scripts.experiments.serve_q4_residual',
                        'compensated-prefill-fp32':'scripts.experiments.serve_prefill_f32',
                        'residual-int8-v2':'scripts.experiments.serve_residual',
                        'residual-int8-fp32':'scripts.experiments.serve_residual_f32'}.get(engine,
                        'scripts.experiments.serve_float32' if precision=='float32' else 'dllm')
                if args.production_residual_f32 and (args.production_prefill_f32 or not (args.prefill_f32 and engine==counterpart)):module='dllm'
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
                process=subprocess.Popen(command,env=phase_environment,stdout=log,stderr=subprocess.STDOUT)
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
                    if args.lifecycle and engine!='llama.cpp':
                        def post(route,body):
                            return urlopen(Request(base+route,data=json.dumps(body).encode(),
                                headers={'Content-Type':'application/json'}),timeout=300)
                        def json_post(route,body):
                            with post(route,body) as response:return json.load(response)
                        body=dict(model='a',prompt=prompts[129],temperature=0,max_tokens=16)
                        expected=next(c for c in run['cases'] if c['prompt_tokens']==129 and c['concurrency']==1)['rounds'][3]['requests'][0]['text']
                        checks={};run['lifecycle']=checks
                        json_post('/v1/models/unload',dict(model='a'))
                        restored=json_post('/v1/completions',body)
                        assert restored['dllm_timing']['source_tier']=='host' and restored['choices'][0]['text']==expected
                        run['kernel_backend']=restored['dllm_timing']['kernel_backend']
                        if args.production_prefill_f32:
                            assert ('compensated-kquant-prefill-f32' in run['kernel_backend']) == (engine==counterpart)
                        checks['host_restore']=True
                        json_post('/v1/models/register',dict(model='b',path=str(swap_model),quantization='none'))
                        swap_body={**body,'model':'b'}
                        if args.swap_model:swap_body['prompt']='Explain binary search in one sentence.'
                        other=json_post('/v1/completions',swap_body)
                        restored=json_post('/v1/completions',body)
                        if not args.swap_model:assert other['choices'][0]['text']==expected
                        assert other['usage']['completion_tokens']>0 and restored['choices'][0]['text']==expected
                        assert restored['dllm_timing']['source_tier']=='host'
                        checks['model_swap' if args.swap_model else 'alias_swap']=True
                        if args.swap_model:checks['swap_sample']=other['choices'][0]['text']
                        json_post('/v1/models/unload',dict(model='a',release_host=True))
                        restored=json_post('/v1/completions',body)
                        assert restored['dllm_timing']['source_tier']=='disk' and restored['choices'][0]['text']==expected
                        checks['disk_reload']=True
                        with post('/v1/completions',{**body,'stream':True,'max_tokens':256}) as response:
                            next(events(response))
                        recovered=json_post('/v1/completions',body)
                        assert recovered['choices'][0]['text']==expected
                        checks['cancellation_recovery']=True
                        chat_body=dict(model='a',messages=[dict(role='user',content='Explain binary search in one short sentence.')],
                                       temperature=0,max_tokens=32)
                        chat=json_post('/v1/chat/completions',chat_body)
                        chat_text=chat['choices'][0]['message']['content']
                        with post('/v1/chat/completions',{**chat_body,'stream':True}) as response:
                            chunks=list(events(response))
                        assert chunks[-1]=='[DONE]'
                        streamed=''.join(c.get('delta',{}).get('content','') for item in chunks[:-1]
                                         for c in json.loads(item)['choices'])
                        assert streamed==chat_text
                        with ThreadPoolExecutor(2) as pool:
                            concurrent=list(pool.map(lambda _:json_post('/v1/chat/completions',chat_body),range(2)))
                        assert all(r['choices'][0]['message']['content']==chat_text for r in concurrent)
                        checks['chat_stream_and_concurrency']=True
                        checks['chat_sample']=chat_text
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
