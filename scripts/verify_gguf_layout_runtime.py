"""Qualify graph, paging, batching and chunking against fixed eager GGUF traces.

This checks execution layouts, not independent model accuracy. Run the separate
HF logit verifier first. One executor is resident at a time so 7B Q4 fits an A2.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from dllm.device import device_name, empty_cache
from dllm.gguf_model import GgufWeights
from dllm.runtime import make_executor, require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model'); parser.add_argument('--output', required=True)
    parser.add_argument('--experimental-q4-residual', action='store_true')
    args = parser.parse_args()
    experiment = None
    if args.experimental_q4_residual:
        from experiments.q4_residual import install_validation_dispatch
        experiment = install_validation_dispatch()
    if os.environ.get('DLLM_KV_CACHE_TOKENS'):
        raise ValueError('run with no inherited KV pool override')
    require_gpu(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    source = GgufWeights(args.model, 512, 'none')
    corpus = source.raw_tokens('The library contains books about oceans and mountains. ')
    prompts = [(corpus * ((n+len(corpus)-1)//len(corpus)))[:n] for n in (4, 128, 256, 384)]
    traces = []; checks = []
    report = dict(experiment=experiment, passed=False, device=device_name(), torch=torch.__version__, context=512,
                  model=str(Path(args.model).resolve()), host_bytes=source.bytes, comparisons=checks,
                  scope='layout equivalence to single-request fixed eager; independent model accuracy is separate')
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
            report['backend'] = executor.kernel_backend
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
                if paged:os.environ['DLLM_KV_CACHE_TOKENS'] = '1024'
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
                    for start in range(0, 384, 128):
                        actual = executor.prefill_chunk(prompts[3][start:start+128],0,start,final=start==256)
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
