"""Profile warm-host executor activation separately from checkpoint preparation."""
import argparse
import gc
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys

import torch
from damendaemon.models import load
from damendaemon.runtime import make_executor, require_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--without-profiler', action='store_true')
    parser.add_argument('--cycles', type=int, default=4)
    parser.add_argument('--native-awq-baseline', help='exact trusted earlier native_awq.py for controlled comparison')
    args = parser.parse_args()
    if args.cycles < 2:
        parser.error('at least a warmup and one measured cycle are required')
    require_gpu()
    if args.native_awq_baseline:
        spec = importlib.util.spec_from_file_location('damendaemon.native_awq', args.native_awq_baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    torch.set_num_threads(4)
    started = time.perf_counter()
    source = load(args.model, args.context, 'none')
    report = dict(scope='warm-host executor activation; no disk load or HTTP queue timing',
                  model=args.model, context=args.context, batch_size=args.batch_size,
                  profiled=not args.without_profiler,
                  native_awq_baseline=args.native_awq_baseline,
                  native_awq_source_sha256=hashlib.sha256(Path(args.native_awq_baseline or 'damendaemon/native_awq.py').read_bytes()).hexdigest(),
                  host_preparation_seconds=time.perf_counter()-started, cycles=[])
    prompt = source.raw_tokens('Explain how oceans affect weather.')
    expected = None
    with torch.inference_mode():
        for cycle in range(args.cycles):
            context = nullcontext() if args.without_profiler else torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
            with context as profile:
                started = time.perf_counter()
                executor = make_executor(source, max_batch=args.batch_size)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
            observed = executor.prefill(prompt).argmax(-1).item()
            if expected is None:
                expected = observed
            if observed != expected:
                raise AssertionError('activation changed the greedy prefill result')
            resident = torch.cuda.memory_allocated()
            backend = executor.kernel_backend
            del executor
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            events = sorted((dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                                  total_cpu_us=e.cpu_time_total) for e in ([] if profile is None else profile.key_averages())),
                            key=lambda e:e['self_cpu_us'], reverse=True)
            item = dict(cycle=cycle, warmup=cycle==0, seconds=elapsed, backend=backend,
                        resident_bytes=resident, released_bytes=torch.cuda.memory_allocated(),
                        greedy_token=observed, cpu_events=events)
            report['cycles'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k!='cpu_events'}), flush=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
