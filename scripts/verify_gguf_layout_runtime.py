"""Qualify graph, paging, batching and chunking against fixed eager GGUF traces.

This checks execution layouts, not independent model accuracy. Run the separate
HF logit verifier first. One executor is resident at a time so 7B Q4 fits an A2.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import torch
from dllm.device import device_name, empty_cache
from dllm.gguf_model import GgufWeights
from dllm.runtime import make_executor, require_gpu


def paged_capacity(context, lengths):
    # Each slot owns whole pages; rounding only the combined token count can
    # under-reserve when several requests cross a page boundary together.
    required = sum(((length + 7 + 15)//16)*16 for length in lengths)
    return ((max(2*context, required)+15)//16)*16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model'); parser.add_argument('--output', required=True)
    parser.add_argument('--context',type=int,default=512,help='context capacity, at least 512')
    parser.add_argument('--prompt-lengths', type=int, nargs=4, help='four explicit lengths, each leaving seven decode positions')
    parser.add_argument('--require-resident-cache', action='store_true', help='fail unless normal resident cache selection is active')
    experiments=parser.add_mutually_exclusive_group()
    experiments.add_argument('--experimental-q4-residual', action='store_true')
    experiments.add_argument('--experimental-residual', action='store_true')
    experiments.add_argument('--experimental-expand-tiles', action='store_true')
    experiments.add_argument('--experimental-dense-cache', action='store_true')
    args = parser.parse_args()
    if args.context<512:parser.error('--context must be at least 512')
    lengths = args.prompt_lengths or (4, 128, args.context//2, args.context-128)
    if any(length <= 0 or length + 7 > args.context for length in lengths):
        parser.error('prompt lengths must be positive and leave seven decode positions')
    pool_tokens = paged_capacity(args.context, lengths)
    experiment = None
    if args.experimental_dense_cache:
        from experiments.kquant_dense_cache import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_expand_tiles:
        from experiments.kquant_expand import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_q4_residual:
        from experiments.q4_residual import install_validation_dispatch
        experiment = install_validation_dispatch()
    if args.experimental_residual:
        from experiments.kquant_residual import install_validation_dispatch
        experiment = install_validation_dispatch()
    if os.environ.get('DLLM_KV_CACHE_TOKENS'):
        raise ValueError('run with no inherited KV pool override')
    require_gpu(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    source = GgufWeights(args.model, args.context, 'none')
    corpus = source.raw_tokens('The library contains books about oceans and mountains. ')
    prompts = [(corpus * ((n+len(corpus)-1)//len(corpus)))[:n]
               for n in lengths]
    traces = []; checks = []
    tokenizer_runtime=dict(kind=type(source.tokenizer).__name__)
    if hasattr(source.tokenizer,'native_model'):
        tokenizer_runtime.update(sentencepiece=version('sentencepiece'),protobuf=version('protobuf'))
    report = dict(tokenizer_runtime=tokenizer_runtime, experiment=experiment, passed=False, device=device_name(), torch=torch.__version__, context=args.context,
                  prompt_lengths=[len(ids) for ids in prompts], chunk_size=128,
                  paged_pool_tokens=pool_tokens,
                  model=str(Path(args.model).resolve()), host_bytes=source.bytes, comparisons=checks,
                  scope='layout equivalence to single-request fixed eager; independent model accuracy is separate')
    kquant_path = os.environ.get('DLLM_KQUANT_CUDA_LIBRARY')
    if kquant_path:
        report['native_kquant_library'] = dict(path=kquant_path,sha256=hashlib.sha256(Path(kquant_path).read_bytes()).hexdigest())
    kquant_residual_path = os.environ.get('DLLM_KQUANT_RESIDUAL_LIBRARY')
    if kquant_residual_path:
        report['native_kquant_residual_library'] = dict(path=kquant_residual_path,
            sha256=hashlib.sha256(Path(kquant_residual_path).read_bytes()).hexdigest())
    library = os.environ.get('DLLM_Q4_CUDA_LIBRARY')
    if library:
        report['native_q4_library'] = dict(path=library, sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
    residual_library = os.environ.get('DLLM_Q4_RESIDUAL_LIBRARY')
    if residual_library:
        report['native_q4_residual_library'] = dict(path=residual_library,sha256=hashlib.sha256(Path(residual_library).read_bytes()).hexdigest())
    def compare(actual, expected, **case):
        expected = expected.to(actual.device)
        error = actual.float()-expected.float()
        item = dict(**case, max_absolute=error.abs().max().item(),
                    relative_rms=(error.square().mean()/expected.float().square().mean().clamp_min(1e-12)).sqrt().item(),
                    greedy_equal=torch.equal(actual.argmax(-1),expected.argmax(-1)))
        checks.append(item)
    try:
        with torch.inference_mode():
            executor = make_executor(source, use_graph=False, max_batch=1)
            if args.require_resident_cache and not getattr(executor, 'packed_cache_bytes', 0):
                raise ValueError('normal resident cache was not selected')
            if args.experimental_dense_cache:
                experiment['resident_cache_bytes'] = executor.quant_linear.experimental_dense_cache_bytes
                if not experiment['resident_cache_bytes']:
                    raise ValueError('dense-cache experiment selected no projections')
            report['backend'] = executor.kernel_backend
            report['resident_projection_cache_bytes'] = getattr(executor, 'packed_cache_bytes', 0)
            report['precision'] = dict(matrix=str(executor.compute_dtype), residual=str(executor.residual_dtype))
            for ids in prompts:
                actual = executor.prefill(ids); trace = []
                for step in range(8):
                    trace.append(actual.cpu())
                    if step < 7:actual, _ = executor.decode(actual.argmax(-1).item())
                traces.append(trace)
            del actual, executor
            gc.collect(); empty_cache()
            for paged in (False, True):
                if paged:os.environ['DLLM_KV_CACHE_TOKENS'] = str(pool_tokens)
                else:os.environ.pop('DLLM_KV_CACHE_TOKENS', None)
                for graph in (False, True):
                    print(f'Checking paged={paged}, graph={graph}', flush=True)
                    executor = make_executor(source, use_graph=graph, max_batch=4)
                    for count in (1, 2, 4):
                        slots = list(range(count))
                        actual = executor.prefill_batch(prompts[:count], slots)
                        for step in range(8):
                            expected = torch.cat([traces[i][step] for i in slots])
                            compare(actual, expected, paged=paged, graph=graph, kind=f'batch{count}', step=step)
                            if step < 7:actual, _ = executor.decode_batch(expected.argmax(-1).tolist(), slots)
                        if paged:
                            for slot in slots:executor.release_request(slot)
                    for start in range(0,len(prompts[3]),128):
                        actual = executor.prefill_chunk(prompts[3][start:start+128],0,start,
                                                       final=start+128>=len(prompts[3]))
                    for step in range(8):
                        compare(actual, traces[3][step], paged=paged, graph=graph, kind='chunked', step=step)
                        if step < 7:actual, _ = executor.decode_batch(traces[3][step].argmax(-1).tolist(), [0])
                    if paged:executor.release_request(0)
                    del actual, executor
                    gc.collect(); empty_cache()
        report['passed'] = all(c['max_absolute'] < .5 and c['relative_rms'] < .01 and c['greedy_equal'] for c in checks)
        if not report['passed']:raise AssertionError('GGUF layout equivalence gate failed')
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        os.environ.pop('DLLM_KV_CACHE_TOKENS', None)
        with Path(args.model).open('rb') as stream:report['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(passed=report['passed'], checks=len(checks),
            max_relative_rms=max((c['relative_rms'] for c in checks), default=None))), flush=True)


if __name__ == '__main__':main()
