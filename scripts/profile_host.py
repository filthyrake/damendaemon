"""Separate warm decode dispatch and token readback; profile host call attribution.

Host API durations can include GPU waits. They are not CPU utilization or pure
Python overhead. Use a device profiler alongside this report before migrating
an execution loop or estimating the gain from doing so.
"""
import argparse
import cProfile
import json
import pstats
import statistics
import time

import torch

from dllm.device import device_name, synchronize
from dllm.qwen2 import ModelWeights
from dllm.runtime import make_executor, require_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.batch <= 8:
        parser.error('steps must be positive and batch must be 1..8')
    torch.set_num_threads(4)
    require_gpu()
    model = ModelWeights(args.path, args.context)
    executor = make_executor(model, max_batch=args.batch, use_graph=not args.eager)
    prompt = model.chat_tokens([{'role':'user', 'content':'Explain matrix multiplication in detail.'}])
    if len(prompt) + args.steps > args.context:
        parser.error('prompt and decode steps exceed context')
    slots = list(range(args.batch))

    @torch.inference_mode()
    def run():
        tokens = executor.prefill_batch([prompt]*args.batch, slots).argmax(-1).cpu().tolist()
        synchronize()
        samples = []
        start = time.perf_counter()
        for _ in range(args.steps):
            before = time.perf_counter()
            _, next_ids = executor.decode_batch(tokens, slots)
            dispatched = time.perf_counter()
            tokens = next_ids.cpu().tolist()
            read = time.perf_counter()
            samples.append({'dispatch_seconds':dispatched-before, 'readback_seconds':read-dispatched})
        elapsed = time.perf_counter()-start
        return {'seconds':elapsed, 'aggregate_tokens_per_second':args.steps*args.batch/elapsed,
                'dispatch_median_seconds':statistics.median(s['dispatch_seconds'] for s in samples),
                'readback_median_seconds':statistics.median(s['readback_seconds'] for s in samples)}

    run()  # Compilation and initial allocations are excluded.
    rounds = [run() for _ in range(3)]
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    calls = []
    for (filename, line, function), (primitive, total, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        calls.append({'file':filename,'line':line,'function':function,'calls':total,
                      'self_seconds':own,'cumulative_seconds':cumulative})
    calls.sort(key=lambda row:row['self_seconds'], reverse=True)
    print(json.dumps({'model':args.path,'device':device_name(),'torch':str(torch.__version__),
        'backend':executor.kernel_backend,'context':args.context,'batch':args.batch,'steps':args.steps,
        'cuda_graphs':bool(executor.batch_graphs),'unprofiled_rounds':rounds,'host_calls':calls[:30],
        'interpretation':'Host API time may include GPU waits; not a measurement of pure Python overhead. '
                         'Profiled calls include prefill; unprofiled timing measures decode only.'},indent=2))


if __name__ == '__main__':
    main()
