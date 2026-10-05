"""Same-resident-weight control/candidate timings for isolated packed MoE prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import MethodType
import torch
from dllm.models import load
from dllm.moe import OlmoeExecutor
from dllm.runtime import make_executor, require_gpu
from scripts.experiments.moe_batched_prefill import install_validation_dispatch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--small-prompts', action='store_true')
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    source = load(args.model, 2048, 'q8_0')
    executor = make_executor(source, max_batch=4)
    if not isinstance(executor, OlmoeExecutor) or not executor.grouped_prefill:
        raise ValueError('qualified A2 Q8 OLMoE configuration required')
    control = OlmoeExecutor.prefill_batch
    dispatch = install_validation_dispatch()
    methods = dict(control=control, candidate=OlmoeExecutor.prefill_batch)
    report = dict(passed=False, model=args.model, device=torch.cuda.get_device_name(),
                  context=2048, max_batch=4, quantization=source.quantization,
                  scope='direct packed prefill; same resident weights; control/candidate/candidate/control; excluded warmups',
                  control_sha256=hashlib.sha256(Path('dllm/moe.py').read_bytes()).hexdigest(),
                  dispatch=dispatch, cases=[])
    corpus = source.raw_tokens('The library contains books about oceans and mountains. ')*100
    def check(actual, target):
        value = actual.cpu().float()
        error = value-target
        rms = (error.square().mean(-1)/target.square().mean(-1).clamp_min(1e-12)).sqrt().max().item()
        absolute = error.abs().max().item()
        greedy = torch.equal(value.argmax(-1), target.argmax(-1))
        if not bool(torch.isfinite(value).all()) or rms >= .01 or absolute >= .5 or not greedy:
            raise AssertionError('packed prefill changes reference output beyond established gates')
        return dict(relative_rms=rms, max_absolute=absolute, greedy_equal=greedy)
    try:
        cases = ([([1,1],[3,1]), ([1]*4,[3,1,0,2]), ([4]*4,[3,1,0,2]),
                  ([16]*4,[3,1,0,2]), ([1,31,65],[2,0,3]), ([1,127],[1,0])]
                 if args.small_prompts else [([31,65],[3,1]), ([32]*4,[3,1,0,2]),
                    ([64,64],[2,0]), ([31,31,65],[2,0,3]), ([65,65],[1,0])])
        for lengths, slots in cases:
            prompts = [corpus[index*7:index*7+length] for index, length in enumerate(lengths)]
            case = dict(lengths=lengths, slots=slots, rounds=[])
            report['cases'].append(case)
            target = None
            for kind in ('control', 'candidate'):
                executor.prefill_batch = MethodType(methods[kind], executor)
                actual = executor.prefill_batch(prompts, slots)
                torch.cuda.synchronize()
                if target is None:
                    target = actual.cpu().float()
                else:
                    check(actual, target)
            for kind in ('control', 'candidate', 'candidate', 'control'):
                executor.prefill_batch = MethodType(methods[kind], executor)
                for trial in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actual = executor.prefill_batch(prompts, slots)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    case['rounds'].append(dict(kind=kind, trial=trial, seconds=elapsed, **check(actual,target)))
            print(json.dumps(case), flush=True)
        if not dispatch['selected_calls'] or (not args.small_prompts and not dispatch['fallback_calls']):
            raise AssertionError('both selected and fallback paths must be measured')
        report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
