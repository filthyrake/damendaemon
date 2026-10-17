"""Replay fixed/paged FP32 execution against a separately produced packed baseline.

The baseline is ordinary damendaemon FP32, whose model accuracy is checked separately
against independent HF. This isolates the candidate's arithmetic and graph
behavior without incorrectly rounding its comparator's K-quant weights to FP16.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from damendaemon.device import device_name, empty_cache
from damendaemon.gguf_model import GgufWeights
from damendaemon.runtime import make_executor, require_gpu


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write-reference');mode.add_argument('--reference')
    experiments=parser.add_mutually_exclusive_group()
    experiments.add_argument('--experimental-residual-f32',action='store_true')
    experiments.add_argument('--experimental-prefill-f32',action='store_true')
    parser.add_argument('--long-prefill',action='store_true',help='include 128/256/384-token prompts and 128-token chunks')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if args.write_reference and (args.experimental_residual_f32 or args.experimental_prefill_f32 or os.environ.get('DAMENDAEMON_KQUANT_RESIDUAL_F32_LIBRARY') or os.environ.get('DAMENDAEMON_KQUANT_PREFILL_F32')=='1'):
        parser.error('the reference must use ordinary packed FP32 dispatch')
    require_gpu();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    experiment=None
    if args.experimental_residual_f32:
        from experiments.kquant_residual_f32 import install_validation_dispatch
        experiment=install_validation_dispatch()
    if args.experimental_prefill_f32:
        from experiments.kquant_prefill_f32 import install_validation_dispatch
        experiment=install_validation_dispatch()
    source=GgufWeights(args.model,512,'none')
    metadata=dict(recipe=2 if args.long_prefill else 1,model_sha256=digest(args.model),context=512,precision='float32',
                  model_type=source.config['model_type'])
    report=dict(**metadata,device=device_name(),torch=torch.__version__,experiment=experiment,
                mode='compare' if args.reference else 'write ordinary packed FP32 baseline',comparisons=[],passed=False)
    production_library=os.environ.get('DAMENDAEMON_KQUANT_RESIDUAL_F32_LIBRARY')
    if production_library:
        report['production_library']=dict(path=production_library,sha256=digest(production_library))
    if os.environ.get('DAMENDAEMON_KQUANT_PREFILL_F32')=='1':
        from damendaemon import kquant_prefill_kernels
        report['production_prefill_kernel_sha256']=digest(kquant_prefill_kernels.__file__)
    reference=np.load(args.reference,allow_pickle=False) if args.reference else None
    saved={};visited=set()
    try:
        if reference is not None:
            if json.loads(str(reference['metadata'].item()))!=metadata:
                raise ValueError('reference model, precision or recipe differs')
            report['reference_sha256']=digest(args.reference)
        def check(actual,*,paged,graph,kind,step):
            key=f'paged{int(paged)}_graph{int(graph)}_{kind}_step{step}'
            observed=actual.float().cpu().numpy()
            if not np.isfinite(observed).all():raise AssertionError(f'non-finite output: {key}')
            expected=reference[key] if reference is not None else observed
            if expected.dtype!=np.float32 or expected.shape!=observed.shape:
                raise ValueError(f'reference shape/dtype differs: {key}')
            error=observed.astype(np.float64)-expected.astype(np.float64)
            relative=float(np.sqrt(np.mean(error**2)/max(np.mean(expected.astype(np.float64)**2),1e-12)))
            absolute=float(np.abs(error).max())
            greedy=np.array_equal(observed.argmax(-1),expected.argmax(-1))
            report['comparisons'].append(dict(key=key,max_absolute=absolute,relative_rms=relative,greedy_equal=greedy))
            visited.add(key)
            if reference is None:saved[key]=observed.copy()
            return expected.argmax(-1).tolist()
        with torch.inference_mode():
            for paged in (False,True):
                if paged:os.environ['DAMENDAEMON_KV_CACHE_TOKENS']='1024'
                else:os.environ.pop('DAMENDAEMON_KV_CACHE_TOKENS',None)
                for graph in (False,True):
                    executor=make_executor(source,use_graph=graph,max_batch=4)
                    if executor.compute_dtype!=torch.float32:raise ValueError('select an FP32 executor policy')
                    report['backend']=executor.kernel_backend
                    prompts=[[9707,11,1879,0]*size for size in ((1,32,64,96) if args.long_prefill else (1,4,8,16))]
                    for count in (1,2,4):
                        slots=list(range(count));actual=executor.prefill_batch(prompts[:count],slots)
                        for step in range(8):
                            tokens=check(actual,paged=paged,graph=graph,kind=f'batch{count}',step=step)
                            if step<7:actual,_=executor.decode_batch(tokens,slots)
                    if args.long_prefill and paged:
                        # Prior requests finished; release their pages before the independent chunk case.
                        for slot in range(4):executor.release_request(slot)
                    chunk=128 if args.long_prefill else 32
                    total=384 if args.long_prefill else 128
                    for start in range(0,total,chunk):
                        actual=executor.prefill_chunk([9707,11,1879,0]*(chunk//4),0,start,final=start==total-chunk)
                    check(actual,paged=paged,graph=graph,kind='chunked',step=0)
                    del actual,executor
                    gc.collect();empty_cache()
        if reference is not None and visited!=set(reference.files)-{'metadata'}:
            raise ValueError('reference cases differ from executed cases')
        report['passed']=all(c['max_absolute']<.5 and c['relative_rms']<.01 and c['greedy_equal']
                             for c in report['comparisons'])
        if args.write_reference:
            saved['metadata']=np.asarray(json.dumps(metadata,sort_keys=True))
            with Path(args.write_reference).open('wb') as handle:np.savez_compressed(handle,**saved)
            report['reference_sha256']=digest(args.write_reference)
        if not report['passed']:raise AssertionError('FP32 packed replay gate failed')
    except Exception as error:
        report['passed']=False;report['error']=repr(error)
        raise
    finally:
        if reference is not None:reference.close()
        os.environ.pop('DAMENDAEMON_KV_CACHE_TOKENS',None)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
