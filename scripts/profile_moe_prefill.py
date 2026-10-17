"""Compare stable OLMoE prefill routing with an exact earlier implementation."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
from types import MethodType

import torch
from damendaemon.models import load
from damendaemon.runtime import make_executor, require_gpu
from damendaemon.moe import OlmoeExecutor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--baseline-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[32, 128, 512, 1024])
    parser.add_argument('--comparison', choices=('bitwise', 'numerical'), default='bitwise')
    parser.add_argument('--quantization', choices=('q8_0', 'q4_0'), default='q8_0',
                        help='Q4 OLMoE requires the explicit experimental runner')
    parser.add_argument('--repeated-token',type=int,
                        help='use a concentrated repeated-token prompt instead of the varied corpus')
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('damendaemon.moe_baseline', args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    source = load(args.model, 2048, args.quantization)
    executor = make_executor(source, max_batch=4)
    if not isinstance(executor, OlmoeExecutor):
        raise ValueError('benchmark requires OLMoE')
    methods = dict(control=baseline.OlmoeExecutor.experts, candidate=OlmoeExecutor.experts)
    report = dict(passed=False, model=args.model, device=torch.cuda.get_device_name(),
                  backend=executor.kernel_backend, quantization=source.quantization,
                  scope='same resident weights and prompts; control/candidate/candidate/control; prefill only',
                  comparison=args.comparison,
                  baseline_sha256=hashlib.sha256(Path(args.baseline_source).read_bytes()).hexdigest(),
                  candidate_source=inspect.getsourcefile(methods['candidate']),
                  candidate_sha256=hashlib.sha256(Path(inspect.getsourcefile(methods['candidate'])).read_bytes()).hexdigest(),
                  cases=[])
    if args.repeated_token is not None:
        if not 0 <= args.repeated_token < source.config['vocab_size']:
            raise ValueError('repeated token is outside the vocabulary')
        corpus = [args.repeated_token]
    else:
        corpus = source.raw_tokens('The library contains books about oceans and mountains. ')
    report['prompt_kind'] = 'varied corpus' if args.repeated_token is None else 'repeated token'
    report['corpus_token_ids'] = corpus
    def check(actual, target):
        value = actual.cpu()
        error = value.float()-target.float()
        rms = (error.square().mean()/target.float().square().mean().clamp_min(1e-12)).sqrt().item()
        maximum = error.abs().max().item()
        exact = torch.equal(value, target)
        greedy = torch.equal(value.argmax(-1), target.argmax(-1))
        if args.comparison == 'bitwise':
            torch.testing.assert_close(value, target, rtol=0, atol=0)
        elif not (bool(torch.isfinite(value).all()) and rms < .01 and maximum < .5 and greedy):
            raise AssertionError('prefill numerical comparison failed')
        return dict(bitwise_equal=exact, relative_rms=rms, max_absolute=maximum, greedy_equal=greedy)
    try:
        for length in args.lengths:
            if not 0 < length <= 2048:
                raise ValueError('invalid prompt length')
            prompt = (corpus*((length+len(corpus)-1)//len(corpus)))[:length]
            case = dict(prompt_tokens=length, rounds=[], profiles={}, peak_extra_allocated_bytes={})
            report['cases'].append(case)
            target = None
            for kind in ('control', 'candidate'):
                executor.experts = MethodType(methods[kind], executor)
                actual = executor.prefill(prompt)
                torch.cuda.synchronize()
                if target is None:
                    target = actual.cpu()
                else:
                    check(actual, target)
            for kind in ('control', 'candidate', 'candidate', 'control'):
                executor.experts = MethodType(methods[kind], executor)
                for trial in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actual = executor.prefill(prompt)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    comparison = check(actual, target)
                    case['rounds'].append(dict(kind=kind, trial=trial, seconds=elapsed, **comparison))
            for kind in ('control', 'candidate'):
                executor.experts = MethodType(methods[kind], executor)
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    executor.prefill(prompt)
                    torch.cuda.synchronize()
                case['peak_extra_allocated_bytes'][kind] = torch.cuda.max_memory_allocated()-resident
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
