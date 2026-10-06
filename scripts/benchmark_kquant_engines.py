"""Sequential, same-GGUF HTTP comparison with verified per-slot context."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import math
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
from dllm.gguf_sentencepiece import restore_tokenizer
from dllm.bench import events,llama_context
from dllm.gguf_model import plan


def digest(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    comparison=parser.add_mutually_exclusive_group(required=True)
    comparison.add_argument('--llama-server')
    comparison.add_argument('--prefill-token-budget',type=int,choices=(1024,2048),help='isolated aggregate Q4 Marlin prefill budget versus normal 512-token grouping')
    comparison.add_argument('--q4-small-tiles',action='store_true',help='compare isolated Q4 prompt tiles at 64/129 rows')
    comparison.add_argument('--expand-tiles',action='store_true',help='compare isolated exact-coefficient expansion tiles')
    comparison.add_argument('--dense-cache',action='store_true',help='compare isolated resident FP16 projection caching')
    comparison.add_argument('--batch-wait-ms',type=float,help='compare normal admission with an isolated idle-group coalescing delay')
    comparison.add_argument('--q4-residual-library',help='compare native Q4 decode with the isolated integer-dot candidate')
    comparison.add_argument('--q4-marlin-library',help='compare isolated Q4 Marlin prefill/batched decode; requires identical residual control on both sides')
    comparison.add_argument('--residual-library',help='compare production with the isolated residual-int8 candidate')
    comparison.add_argument('--residual-f32-library',help='compare ordinary packed FP32 with the isolated FP32 candidate')
    parser.add_argument('--production-q4-residual',action='store_true',help='exercise normal per-model Q4 integer-dot selection')
    parser.add_argument('--q4-residual-control-library',help='use this existing native residual library in control phases, instead of ordinary native Q4')
    parser.add_argument('--dllm-q4-marlin-library',help='use normal Q4 Marlin for dllm in a matched llama.cpp comparison')
    parser.add_argument('--production-q4-marlin',action='store_true',help='use normal admission-accounted Q4 Marlin layout replacement')
    parser.add_argument('--q4-marlin-symmetric',action='store_true',help='isolated implicit-zero-point replacement versus qualified Marlin')
    parser.add_argument('--q4-marlin-control-library',help='qualified ABI 1 control for symmetric experiment or production ABI 2 upgrade')
    parser.add_argument('--q4-marlin-replace',action='store_true',help='use the isolated all-row Marlin GPU layout replacement')
    parser.add_argument('--production-q4-small-tiles',action='store_true',help='compare the previous Q4 prompt policy with normal A2 selection')
    parser.add_argument('--production-batch-wait',action='store_true',help='exercise normal CLI idle admission configuration')
    parser.add_argument('--production-residual',action='store_true',help='exercise normal per-model FP16 K-quant residual selection')
    parser.add_argument('--production-dense-cache',action='store_true',help='exercise normal admission-accounted resident cache')
    parser.add_argument('--dense-cache-mib',type=int,default=2048,help='normal resident cache budget per executor')
    parser.add_argument('--production-prefill-f32',action='store_true',help='exercise normal per-executor compensated prefill selection')
    parser.add_argument('--prefill-f32',action='store_true',help='compare production FP32 residual decode with experimental compensated prefill')
    parser.add_argument('--production-residual-f32',action='store_true',help='exercise normal CLI per-executor FP32 selection')
    parser.add_argument('--include-concurrency-four',action='store_true',help='also measure four simultaneous 129-token prompts')
    parser.add_argument('--context',type=int,default=512,help='context capacity per slot, at least 512')
    parser.add_argument('--extra-prompt-tokens',type=int,nargs='*',default=[],help='additional single-request prompt lengths; existing controls remain')
    parser.add_argument('--long-prompt-tokens',type=int,default=400,help='length of the long single-request workload')
    parser.add_argument('--prompt-text',default='Binary search repeatedly halves a sorted range. A detailed explanation follows: ',
                        help='text repeated to construct identical explicit prompt IDs for both engines')
    parser.add_argument('--lifecycle',action='store_true',help='also check dllm cancellation, swapping and reloads after timing')
    parser.add_argument('--swap-model', help='optional different GGUF checkpoint for lifecycle swapping')
    parser.add_argument('--native-format', choices=('kquant','q4'), default='kquant', help='native decode library format')
    parser.add_argument('--native-library',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.context<512:parser.error('--context must be at least 512')
    if any(n < 1 or n + 16 > args.context for n in args.extra_prompt_tokens):
        parser.error('extra prompt lengths must be positive and leave 16 output tokens')
    if not args.prompt_text:parser.error('--prompt-text must be nonempty')
    if not 129<args.long_prompt_tokens<=args.context-16:
        parser.error('--long-prompt-tokens must exceed 129 and leave room for 16 output tokens')
    if args.swap_model and not args.lifecycle:parser.error('--swap-model requires --lifecycle')
    if args.batch_wait_ms is not None and (not math.isfinite(args.batch_wait_ms) or not 0<args.batch_wait_ms<=10):
        parser.error('--batch-wait-ms must be >0 and <=10')
    if args.native_format=='q4' and not (args.llama_server or args.q4_residual_library or args.q4_marlin_library or args.batch_wait_ms or args.q4_small_tiles or args.prefill_token_budget):
        parser.error('--native-format q4 requires a Q4 engine, kernel or admission comparison')
    if args.q4_residual_library and args.native_format!='q4':parser.error('Q4 candidate requires --native-format q4')
    if args.q4_small_tiles and args.native_format!='q4':parser.error('Q4 tiles require --native-format q4')
    if args.production_q4_small_tiles and not args.q4_small_tiles:
        parser.error('--production-q4-small-tiles requires --q4-small-tiles')
    if args.production_q4_residual and not args.q4_residual_library:parser.error('--production-q4-residual requires --q4-residual-library')
    if args.q4_residual_control_library and not (args.production_q4_residual or args.q4_marlin_library):
        parser.error('--q4-residual-control-library requires --production-q4-residual or --q4-marlin-library')
    if args.q4_marlin_library and (args.native_format!='q4' or not args.q4_residual_control_library):
        parser.error('--q4-marlin-library requires --native-format q4 and --q4-residual-control-library')
    if args.dllm_q4_marlin_library and (not (args.llama_server or args.prefill_token_budget) or args.native_format!='q4'):
        parser.error('--dllm-q4-marlin-library requires a llama-server or prefill-budget comparison and --native-format q4')
    if args.prefill_token_budget and (not args.dllm_q4_marlin_library or args.context!=512 or not args.lifecycle):
        parser.error('--prefill-token-budget requires a native Marlin library, context 512 and lifecycle checks')
    if args.q4_marlin_symmetric and (not args.q4_marlin_library or not args.q4_marlin_control_library or args.q4_marlin_replace or args.production_q4_marlin):
        parser.error('--q4-marlin-symmetric requires candidate/control libraries and excludes other Marlin modes')
    if args.q4_marlin_control_library and not (args.q4_marlin_symmetric or args.production_q4_marlin):
        parser.error('--q4-marlin-control-library requires --q4-marlin-symmetric or --production-q4-marlin')
    if args.production_q4_marlin and (not args.q4_marlin_library or args.q4_marlin_replace):
        parser.error('--production-q4-marlin requires --q4-marlin-library and excludes --q4-marlin-replace')
    if args.q4_marlin_replace and not args.q4_marlin_library:
        parser.error('--q4-marlin-replace requires --q4-marlin-library')
    if args.production_batch_wait and not args.batch_wait_ms:parser.error('--production-batch-wait requires --batch-wait-ms')
    if args.production_residual and not args.residual_library:parser.error('--production-residual requires --residual-library')
    if args.production_dense_cache and (not args.dense_cache or args.dense_cache_mib <= 0):
        parser.error('--production-dense-cache requires --dense-cache and a positive budget')
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
    source=plan(model);tokenizer=restore_tokenizer(source.tokenizer_json, source.sentencepiece_prefix)
    if precision == 'float32' and source.config['model_type'] not in ('llama', 'qwen3'):
        parser.error('this benchmark can select FP32 only for Llama and Qwen3')
    precision_description = ('FP32 matrices, residual/nonlinear and KV' if precision == 'float32' else
                             'mixed: FP16 matrices, FP32 residual/nonlinear' if source.config['model_type'] in ('llama', 'qwen3') else
                             'standard: FP16 matrices, residual and KV')
    base_tokens=tokenizer.encode(args.prompt_text,add_special_tokens=True).ids
    prompts={length:(base_tokens*((length+len(base_tokens)-1)//len(base_tokens)))[:length]
             for length in (16,129,args.long_prompt_tokens,*args.extra_prompt_tokens)}
    tokenizer_runtime=dict(kind=type(tokenizer).__name__)
    if hasattr(tokenizer,'native_model'):
        tokenizer_runtime.update(sentencepiece=version('sentencepiece'),protobuf=version('protobuf'))
    report=dict(tokenizer_runtime=tokenizer_runtime, native_format=args.native_format, model=str(model),model_sha256=digest(model),native_library=str(library),
        native_library_sha256=digest(library),llama_server=str(llama) if llama else None,llama_server_sha256=digest(llama) if llama else None,
        residual_library=str(residual) if residual else None,
        residual_library_sha256=digest(residual) if residual else None,
        experimental_arithmetic=('two dyadic int8 activation terms; FP32 rows 1-4, widths 2048/8192' if precision=='float32'
                                else 'two dyadic int8 activation terms; FP16 single-row width-2048 K-quants') if residual else None,
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],context_per_slot=args.context,slots=4,kv_dtype='float32' if precision=='float32' else 'float16',
        dllm_precision=precision_description,model_type=source.config['model_type'],output_tokens=16,
        prompts=prompts,prompt_text=args.prompt_text,warmup_rounds=3,measured_rounds=6,runs=[])
    if args.q4_residual_library:
        report['experimental_arithmetic']='Q4 single-row integer dot; two dyadic int8 activation terms; FP32 coefficient scaling rather than rounded FP16 coefficients'
    report['swap_model']=dict(path=str(swap_model),sha256=digest(swap_model)) if args.swap_model else None
    report['experimental_batch_wait_ms']=args.batch_wait_ms if not args.production_batch_wait else None
    report['production_batch_wait_ms']=args.batch_wait_ms if args.production_batch_wait else None
    workloads=[(16,1),(129,1),(args.long_prompt_tokens,1),(129,2)]
    if args.include_concurrency_four:workloads.append((129,4))
    workloads.extend((n,1) for n in dict.fromkeys(args.extra_prompt_tokens) if (n,1) not in workloads)
    report['workloads']=workloads
    report['production_q4_residual']=args.production_q4_residual
    if args.q4_residual_control_library:
        control=Path(args.q4_residual_control_library).resolve()
        report['q4_residual_control_library']=dict(path=str(control),sha256=digest(control))
        report['experimental_arithmetic']='normal Q4 residual dispatch in both phases; explicit control and candidate libraries'
    if args.q4_marlin_library:
        marlin=Path(args.q4_marlin_library).resolve()
        report['q4_marlin_library']=dict(path=str(marlin),sha256=digest(marlin))
        report['experimental_arithmetic']='same single-row residual library; isolated group-32 Marlin for rows >1 over the same Q4 coefficients/scales'
        report['production_admission_accounts_for_extra_weights']=False
        report['q4_marlin_dispatch_sha256']=digest(Path(__file__).parent/'experiments/q4_marlin_dispatch.py')
        report['q4_marlin_adapter_sha256']=digest(Path(__file__).parent/'experiments/q4_marlin.py')
        report['q4_marlin_replaces_gpu_layout']=args.q4_marlin_replace or args.q4_marlin_symmetric
        if args.q4_marlin_replace:
            report['experimental_arithmetic']='all-row Marlin over the same Q4 coefficients/scales, rounded FP16 coefficients; control uses qualified residual single-row FP32 coefficient scaling'
            report['q4_marlin_replacement_sources']={name:digest(Path(__file__).parent/'experiments'/name)
                for name in ('q4_marlin_replace.py','q4_marlin_direct.py','q4_marlin_repack.py')}
    if args.q4_marlin_control_library:
        marlin_control=Path(args.q4_marlin_control_library).resolve()
        report['q4_marlin_control_library']=dict(path=str(marlin_control),sha256=digest(marlin_control))
    if args.q4_marlin_symmetric:
        report['experimental_arithmetic']='implicit zero point eight versus qualified explicit-zero-point Marlin; same FP16 coefficients and scales'
        report['implicit_zero_point']=8
        report['symmetric_sources']={name:digest(Path(__file__).parent/'experiments'/name) for name in
            ('q4_marlin_symmetric.py','q4_marlin_symmetric.cu','q4_marlin_replace.py','q4_marlin_direct.py','q4_marlin_repack.py')}
    if args.dllm_q4_marlin_library:
        dllm_marlin=Path(args.dllm_q4_marlin_library).resolve()
        report['dllm_q4_marlin_library']=dict(path=str(dllm_marlin),sha256=digest(dllm_marlin))
        report['experimental_arithmetic']='normal Q4 Marlin FP16 coefficients and mixed residuals versus llama.cpp over identical GGUF bytes; engine arithmetic differs'
    report['production_q4_marlin']=args.production_q4_marlin
    if args.production_q4_marlin:
        report['production_admission_accounts_for_extra_weights']=True
        report['q4_marlin_replaces_gpu_layout']=True
        report['experimental_arithmetic']='normal all-row Marlin replacement with FP16 coefficients; residual control uses FP32 scaling'
        if args.q4_marlin_control_library:
            report['experimental_arithmetic']='normal ABI 2 implicit zero point eight versus qualified ABI 1 explicit zero points; same FP16 coefficients and scales'
        report['q4_marlin_production_sources']={name:digest(Path(__file__).resolve().parents[1]/name)
            for name in ('dllm/native_q4_marlin.py','dllm/q4_marlin_repack.py','dllm/q4_marlin_policy.py','native/q4_marlin.cu')}
    report['production_residual_f32']=args.production_residual_f32
    report['experimental_prefill_f32']=args.prefill_f32 and not args.production_prefill_f32
    report['production_prefill_f32']=args.production_prefill_f32
    if args.prefill_f32:
        report['experimental_arithmetic']='FP32 prefill rows >=128; scaled high/low FP16 operands, three products; same production residual decode in both engines'
        report['prefill_source_sha256']=digest(Path(__file__).parent/'experiments/kquant_prefill_f32.py')
        report['prefill_kernel_sha256']=digest(Path(__file__).resolve().parents[1]/'dllm/kquant_prefill_kernels.py')
    environment=os.environ.copy()
    environment.pop('DLLM_KQUANT_DENSE_CACHE_MIB',None)
    environment.pop('DLLM_KQUANT_RESIDUAL_F32_LIBRARY',None)
    environment.pop('DLLM_KQUANT_PREFILL_F32',None)
    environment.pop('DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY',None)
    environment.pop('DLLM_Q4_MARLIN_LIBRARY',None)
    environment.pop('DLLM_EXPERIMENTAL_Q4_MARLIN_LIBRARY',None)
    environment.pop('DLLM_EXPERIMENTAL_Q4_MARLIN_REPLACE',None)
    environment.pop('DLLM_EXPERIMENTAL_Q4_MARLIN_SYMMETRIC',None)
    environment.pop('DLLM_EXPERIMENTAL_PREFILL_TOKEN_BUDGET',None)
    environment.pop('DLLM_Q4_RESIDUAL_LIBRARY',None)
    environment.pop('DLLM_EXPERIMENTAL_BATCH_WAIT_MS',None)
    environment.pop('DLLM_Q4_CUDA_LIBRARY',None)
    environment.pop('DLLM_KQUANT_CUDA_LIBRARY',None)
    environment['DLLM_Q4_CUDA_LIBRARY' if args.native_format=='q4' else 'DLLM_KQUANT_CUDA_LIBRARY']=str(library)
    environment.update(DLLM_DEVICE='cuda',DLLM_LLAMA_PRECISION=precision,
        DLLM_QWEN3_PRECISION='float32' if precision=='float32' else 'standard',
        DLLM_SPECULATIVE_TOKENS='0',DLLM_QUANTIZATION='none',OMP_NUM_THREADS='4')
    production_residual=os.environ.get('DLLM_KQUANT_RESIDUAL_LIBRARY')
    if args.production_residual:
        environment.pop('DLLM_KQUANT_RESIDUAL_LIBRARY',None)
        production_residual=None
    if production_residual:
        report['production_residual_library']=dict(path=production_residual,sha256=digest(production_residual))
    report['experimental_batch2_q6']=bool(args.residual_library and os.environ.get('DLLM_EXPERIMENTAL_RESIDUAL_BATCH2')=='1')
    if report['experimental_batch2_q6']:
        report['experimental_arithmetic']+='; batch-two pure Q6 at (width, outputs)=(8192,2048) or (2048,128256)'
    counterpart='residual-int8-fp32' if precision=='float32' else 'residual-int8-v2' if residual else 'llama.cpp'
    if args.q4_residual_library:counterpart='q4-residual-int8'
    if args.q4_marlin_library:counterpart='q4-marlin-replaced' if args.q4_marlin_replace else 'q4-marlin'
    if args.production_q4_marlin:counterpart='q4-marlin-production'
    if args.q4_marlin_symmetric:counterpart='q4-marlin-symmetric'
    if args.prefill_token_budget:
        counterpart='prefill-budget'
        report['experimental_prefill_token_budget']=args.prefill_token_budget
        report['experimental_arithmetic']='same Q4 Marlin ABI 2 coefficients; aggregate prefill grouping only; per-slot context remains 512'
        report['prefill_budget_source_sha256']=digest(Path(__file__).parent/'experiments/q4_marlin_prefill_budget.py')
    if args.q4_small_tiles:
        counterpart='q4-small-tiles'
        report['experimental_arithmetic']='unchanged FP16 coefficients; paired-dot 32-row tiles at 64/129 rows'
        report['tile_source_sha256']=digest(Path(__file__).parent/'experiments/q4_small_tiles.py')
        report['production_q4_small_tiles']=args.production_q4_small_tiles
        if args.production_q4_small_tiles:
            report['experimental_arithmetic']='unchanged FP16 coefficients; normal A2 33-64-row policy versus previous prompt tiles'
            report['tile_source_sha256']=digest(Path(__file__).resolve().parents[1]/'dllm/q4_prefill_policy.py')
    if args.batch_wait_ms:counterpart='coalesced-prefill'
    if args.expand_tiles:
        counterpart='expand-tiles'
        report['experimental_arithmetic']='unchanged FP16 coefficients; 8x512 expansion tiles, four warps'
        report['expansion_source_sha256']=digest(Path(__file__).parent/'experiments/kquant_expand.py')
        report['expansion_kernel_sha256']=digest(Path(__file__).resolve().parents[1]/'dllm/kquant_kernels.py')
    if args.dense_cache:
        counterpart='dense-cache'
        report['experimental_arithmetic']='same FP16 coefficients; resident expanded projections for rows >1; extra VRAM, experimental accounting only'
        report['dense_cache_source_sha256']=digest(Path(__file__).parent/'experiments/kquant_dense_cache.py')
        report['production_dense_cache']=args.production_dense_cache
        if args.production_dense_cache:
            report['experimental_arithmetic']='same FP16 coefficients; normal admission-accounted resident projection cache'
            report['dense_cache_source_sha256']=digest(Path(__file__).resolve().parents[1]/'dllm/kquant_cache.py')
            report['dense_cache_budget_mib']=args.dense_cache_mib
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
            if args.dllm_q4_marlin_library and (engine=='dllm' or args.prefill_token_budget):
                phase_environment['DLLM_Q4_MARLIN_LIBRARY']=str(dllm_marlin)
            if args.prefill_token_budget and engine==counterpart:
                phase_environment['DLLM_EXPERIMENTAL_PREFILL_TOKEN_BUDGET']=str(args.prefill_token_budget)
            if args.q4_marlin_control_library and engine=='dllm':
                phase_environment['DLLM_Q4_MARLIN_LIBRARY']=str(marlin_control)
            if args.q4_residual_control_library:
                phase_environment['DLLM_Q4_RESIDUAL_LIBRARY']=str(control)
            if args.q4_marlin_library and engine==counterpart:
                phase_environment['DLLM_Q4_MARLIN_LIBRARY' if args.production_q4_marlin else 'DLLM_EXPERIMENTAL_Q4_MARLIN_LIBRARY']=str(marlin)
                if args.q4_marlin_replace:phase_environment['DLLM_EXPERIMENTAL_Q4_MARLIN_REPLACE']='1'
                if args.q4_marlin_symmetric:phase_environment['DLLM_EXPERIMENTAL_Q4_MARLIN_SYMMETRIC']='1'
            if args.production_dense_cache and engine==counterpart:
                phase_environment['DLLM_KQUANT_DENSE_CACHE_MIB']=str(args.dense_cache_mib)
            if args.batch_wait_ms and engine==counterpart and not args.production_batch_wait:
                phase_environment['DLLM_EXPERIMENTAL_BATCH_WAIT_MS']=str(args.batch_wait_ms)
            if args.production_residual and engine==counterpart:
                phase_environment['DLLM_KQUANT_RESIDUAL_LIBRARY']=str(residual)
            if args.q4_residual_library and engine==counterpart:
                phase_environment['DLLM_Q4_RESIDUAL_LIBRARY' if args.production_q4_residual else 'DLLM_EXPERIMENTAL_Q4_RESIDUAL_LIBRARY']=str(residual)
            if args.production_residual_f32 and engine==counterpart:
                phase_environment['DLLM_KQUANT_RESIDUAL_F32_LIBRARY']=str(residual)
            if args.production_prefill_f32 and engine==counterpart:
                phase_environment['DLLM_KQUANT_PREFILL_F32']='1'
            if engine!='llama.cpp':
                module={'prefill-budget':'scripts.experiments.serve_prefill_budget',
                        'coalesced-prefill':'scripts.experiments.serve_batch_wait',
                        'q4-small-tiles':'scripts.experiments.serve_q4_small_tiles',
                        'expand-tiles':'scripts.experiments.serve_expand_tiles',
                        'dense-cache':'scripts.experiments.serve_dense_cache',
                        'q4-residual-int8':'scripts.experiments.serve_q4_residual',
                        'q4-marlin':'scripts.experiments.serve_q4_marlin',
                        'q4-marlin-replaced':'scripts.experiments.serve_q4_marlin',
                        'q4-marlin-symmetric':'scripts.experiments.serve_q4_marlin',
                        'compensated-prefill-fp32':'scripts.experiments.serve_prefill_f32',
                        'residual-int8-v2':'scripts.experiments.serve_residual',
                        'residual-int8-fp32':'scripts.experiments.serve_residual_f32'}.get(engine,
                        'scripts.experiments.serve_float32' if precision=='float32' else 'dllm')
                if args.production_q4_residual or args.production_residual or args.production_batch_wait:module='dllm'
                if args.production_dense_cache:module='dllm'
                if args.production_q4_small_tiles:
                    module='dllm' if engine==counterpart else 'scripts.experiments.serve_q4_legacy_tiles'
                if args.production_residual_f32 and (args.production_prefill_f32 or not (args.prefill_f32 and engine==counterpart)):module='dllm'
                command=[sys.executable,'-m',module,'serve','--model',f'a={model}','--port',str(port),
                    '--context',str(args.context),'--batch-size','4','--kv-cache-tokens','0','--gpu-models','1',
                    '--prefill-chunk-size','0','--prefix-cache-tokens','0']
                if args.production_batch_wait:
                    command+=['--batch-wait-ms',str(args.batch_wait_ms if engine==counterpart else 0)]
            else:
                command=[str(llama),'--model',str(model),'--alias','a','--host','127.0.0.1','--port',str(port),
                    '--ctx-size',str(4*args.context),'--parallel','4','--kv-unified-per-slot',str(args.context),'--gpu-layers','99',
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
                        run['context']=llama_context(base,args.context)
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
                    for length,concurrency in workloads:
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
                    if args.q4_marlin_library or (args.dllm_q4_marlin_library and (engine=='dllm' or args.prefill_token_budget)):
                        with urlopen(base+'/v1/models/status',timeout=30) as response:
                            run['memory_after_workloads']=json.load(response)
                    if args.lifecycle and engine!='llama.cpp':
                        def post(route,body):
                            return urlopen(Request(base+route,data=json.dumps(body).encode(),
                                headers={'Content-Type':'application/json'}),timeout=300)
                        def json_post(route,body,timing=None):
                            started=time.perf_counter()
                            with post(route,body) as response:value=json.load(response)
                            if timing:
                                run.setdefault('lifecycle_timings',{})[timing]=dict(
                                    seconds=time.perf_counter()-started,dllm_timing=value.get('dllm_timing'))
                            return value
                        body=dict(model='a',prompt=prompts[129],temperature=0,max_tokens=16)
                        expected=next(c for c in run['cases'] if c['prompt_tokens']==129 and c['concurrency']==1)['rounds'][3]['requests'][0]['text']
                        checks={};run['lifecycle']=checks
                        json_post('/v1/models/unload',dict(model='a'))
                        restored=json_post('/v1/completions',body,timing='host_restore_request')
                        assert restored['dllm_timing']['source_tier']=='host' and restored['choices'][0]['text']==expected
                        run['kernel_backend']=restored['dllm_timing']['kernel_backend']
                        if args.production_dense_cache:
                            assert ('+resident-kquant-cache' in run['kernel_backend']) == (engine==counterpart)
                            with urlopen(base+'/v1/models/status',timeout=30) as response:
                                status=json.load(response)
                            cache_bytes=next(m['gpu_projection_cache_bytes'] for m in status['models'] if m['id']=='a')
                            assert (0 < cache_bytes <= args.dense_cache_mib*1024**2) if engine==counterpart else cache_bytes==0
                            run['resident_projection_cache_bytes']=cache_bytes
                        if args.production_q4_residual:
                            assert ('+cuda-q4-residual' in run['kernel_backend']) == (engine==counterpart or bool(args.q4_residual_control_library))
                        if args.dllm_q4_marlin_library:
                            assert '+cuda-q4-marlin' in run['kernel_backend']
                            selected=run['memory_after_workloads']['models'][0]
                            assert selected['q4_marlin_abi'] in (1,2)
                            assert selected['gpu_packed_layout_overhead_bytes']==(111099904 if selected['q4_marlin_abi']==1 else 0)
                        if args.prefill_token_budget:
                            assert selected['q4_marlin_abi']==2
                            marker=f'+experimental-prefill-budget{args.prefill_token_budget}'
                            assert (marker in run['kernel_backend']) == (engine==counterpart)
                        if args.production_q4_marlin:
                            native_selected=engine==counterpart or bool(args.q4_marlin_control_library)
                            assert ('+cuda-q4-marlin' in run['kernel_backend']) == native_selected
                            assert ('+cuda-q4-residual' in run['kernel_backend']) == (not native_selected)
                            selected=run['memory_after_workloads']['models'][0]
                            if native_selected:
                                assert selected['q4_marlin_abi'] in (1,2)
                                if args.q4_marlin_control_library:
                                    assert selected['q4_marlin_abi']==(2 if engine==counterpart else 1)
                            else:assert selected['q4_marlin_abi'] is None
                            overhead=selected['gpu_packed_layout_overhead_bytes']
                            assert overhead == (111099904 if selected['q4_marlin_abi']==1 else 0),overhead
                        if args.q4_marlin_symmetric:
                            assert '+cuda-q4-residual' not in run['kernel_backend']
                            assert ('+cuda-q4-marlin' in run['kernel_backend']) == (engine=='dllm')
                            assert ('+experimental-q4-marlin-symmetric' in run['kernel_backend']) == (engine==counterpart)
                        if args.q4_marlin_library and not (args.production_q4_marlin or args.q4_marlin_symmetric):
                            assert ('+cuda-q4-residual' in run['kernel_backend']) == (not args.q4_marlin_replace or engine!=counterpart)
                            assert ('+experimental-q4-marlin' in run['kernel_backend']) == (engine==counterpart)
                            assert ('+experimental-q4-marlin-replaced' in run['kernel_backend']) == (args.q4_marlin_replace and engine==counterpart)
                        if args.production_residual:
                            assert ('+cuda-kquant-residual' in run['kernel_backend']) == (engine==counterpart)
                        if args.production_prefill_f32:
                            assert ('compensated-kquant-prefill-f32' in run['kernel_backend']) == (engine==counterpart)
                        checks['host_restore']=True
                        json_post('/v1/models/register',dict(model='b',path=str(swap_model),quantization='none'))
                        swap_body={**body,'model':'b'}
                        if args.swap_model:swap_body['prompt']='Explain binary search in one sentence.'
                        other=json_post('/v1/completions',swap_body,timing='switch_to_other_request')
                        restored=json_post('/v1/completions',body,timing='switch_back_request')
                        if not args.swap_model:assert other['choices'][0]['text']==expected
                        assert other['usage']['completion_tokens']>0 and restored['choices'][0]['text']==expected
                        assert restored['dllm_timing']['source_tier']=='host'
                        checks['model_swap' if args.swap_model else 'alias_swap']=True
                        if args.swap_model:checks['swap_sample']=other['choices'][0]['text']
                        json_post('/v1/models/unload',dict(model='a',release_host=True))
                        restored=json_post('/v1/completions',body,timing='disk_reload_request')
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
