"""Compare stable OLMoE prefill routing with an exact earlier implementation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import MethodType

import torch
from dllm.models import load
from dllm.runtime import make_executor, require_gpu
from dllm.moe import OlmoeExecutor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[32, 128, 512, 1024])
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('dllm.moe_baseline', args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    source = load(args.model, 2048, 'q8_0')
    executor = make_executor(source, max_batch=4)
    if not isinstance(executor, OlmoeExecutor):
        raise ValueError('benchmark requires OLMoE')
    methods = dict(control=baseline.OlmoeExecutor.experts, candidate=OlmoeExecutor.experts)
    report = dict(passed=False, model=args.model, device=torch.cuda.get_device_name(),
                  backend=executor.kernel_backend, quantization=source.quantization,
                  scope='same resident weights and prompts; control/candidate/candidate/control; prefill only',
                  baseline_sha256=hashlib.sha256(Path(args.baseline_source).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(Path('dllm/moe.py').read_bytes()).hexdigest(),
                  cases=[])
    corpus = source.raw_tokens('The library contains books about oceans and mountains. ')
    try:
        for length in args.lengths:
            if not 0 < length <= 2048:
                raise ValueError('invalid prompt length')
            prompt = (corpus*((length+len(corpus)-1)//len(corpus)))[:length]
            case = dict(prompt_tokens=length, rounds=[], profiles={})
            report['cases'].append(case)
            target = None
            for kind in ('control', 'candidate'):
                executor.experts = MethodType(methods[kind], executor)
                actual = executor.prefill(prompt)
                torch.cuda.synchronize()
                if target is None:
                    target = actual.cpu()
                else:
                    torch.testing.assert_close(actual.cpu(), target, rtol=0, atol=0)
            for kind in ('control', 'candidate', 'candidate', 'control'):
                executor.experts = MethodType(methods[kind], executor)
                for trial in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actual = executor.prefill(prompt)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    torch.testing.assert_close(actual.cpu(), target, rtol=0, atol=0)
                    case['rounds'].append(dict(kind=kind, trial=trial, seconds=elapsed, bitwise_equal=True))
            for kind in ('control', 'candidate'):
                executor.experts = MethodType(methods[kind], executor)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    executor.prefill(prompt)
                    torch.cuda.synchronize()
                case['profiles'][kind] = [dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total)
                    for e in profile.key_averages() if any(part in e.key for part in ('nonzero','bincount','sort','Synchronize'))]
            print(json.dumps(case), flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
