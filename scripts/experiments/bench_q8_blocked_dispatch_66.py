"""Thermally paced #66 full-model gate: blocked Q8 matvec dispatch ABBA.

October's compensated-dispatch ABBA harness (bench_hybrid_compensated_dispatch)
with two changes, both recorded in the report:

- Arm B replaces the small-batch FP32 matvec with the frozen blocked
  reduction (scripts/experiments/q8_blocked_matvec.py) for one to four rows;
  arm A is exactly October's arm A (FP32 matvec for rows <=4, compensated
  BF16 prefill for rows >=5). Storage, state precision, histories and the
  84-target numerical gates are unchanged.
- Every ramp and measured phase is thermally paced
  (scripts/experiments/thermal_pacing.py): paced CPU quantization, paced
  residency fill, and a cool-down gate before each timed replay event.
  Pacing inserts idle between already-synchronous operations; the timed
  windows themselves still bracket one synchronize/event/synchronize and
  are therefore comparable with October's unpaced medians.

Run under guard_homer on HOMER (fans fixed at maximum); the pacer's soft
stop must stay under the guard's --limit-celsius.
"""
if not __debug__:
    raise RuntimeError('the #66 gate asserts numerical gates; run Python without -O or -OO.')
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from scripts.experiments import thermal_pacing
from scripts.experiments.q8_blocked_matvec import linear as blocked
from damendaemon.compensated_q8 import linear as compensated
from damendaemon import quantized_kernels
from damendaemon.device import synchronize, device_name
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import make_executor, require_gpu

original_linear = quantized_kernels.linear
blocked_arm = False


def dispatch(x, weight, scales, bias=None):
    rows = x.numel() // x.shape[-1] if x.dim() else 0
    if x.dtype == torch.float32 and bias is None:
        if blocked_arm and 1 <= rows <= 4:
            return blocked(x, weight, scales)
        if rows >= 5:
            return compensated(x, weight, scales)
    return original_linear(x, weight, scales, bias)


@torch.inference_mode()
def replay(executor, events, pacer):
    checks, timings = [], []
    for index, event in enumerate(events):
        pacer.gate('event')
        synchronize()
        start = time.perf_counter()
        if event['kind'] == 'prefill':
            outputs = []
            for prompt, slot in zip(event['prompts'], event['slots']):
                for offset in range(0, len(prompt), 127):
                    result = executor.prefill_chunk(prompt[offset:offset+127], slot, offset,
                                                    final=offset+127 >= len(prompt))
                outputs.append(result)
            actual = torch.cat(outputs)
        else:
            actual, _ = executor.decode_batch(event['tokens'], event['slots'])
        synchronize()
        timings.append(dict(event_index=index, kind=event['kind'], slots=event['slots'],
                            seconds=time.perf_counter()-start))
        for row, target in enumerate(event['targets']):
            value = actual[row].float().cpu()
            target = target.squeeze(0)
            error = value-target
            rms = (error.square().mean().sqrt()
                   / target.square().mean().sqrt().clamp_min(1e-6)).item()
            maximum = error.abs().max().item()
            match = value.argmax().item() == target.argmax().item()
            checks.append(dict(event_index=index, slot=event['slots'][row], relative_rms=rms,
                               max_absolute=maximum, greedy_match=match,
                               passed=rms < .01 and maximum < .5 and match))
    return dict(checks=checks, timings=timings,
                passed=all(row['passed'] for row in checks))


def main():
    global blocked_arm
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    parser.add_argument('--soft-stop', type=float, default=80.0,
                        help='pacer soft stop in Celsius; keep below guard_homer limit')
    parser.add_argument('--poll-seconds', type=float, default=0.5)
    parser.add_argument('--pilot', action='store_true',
                        help='run warmup-A plus one measured A pass only')
    args = parser.parse_args()
    if os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'float32':
        raise RuntimeError('the gate pins FP32 state precision; set DAMENDAEMON_HYBRID_PRECISION=float32')
    require_gpu()
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1151'):
        raise RuntimeError('the blocked-dispatch gate is restricted to the qualified HOMER target')
    torch.set_num_threads(4)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    if (not trace.get('complete', True) or trace['context'] != 512
            or trace['reference_dtype'] != 'float32' or trace['reference_quantization'] != 'q8_0'
            or len(trace['trace']) != 29 or sum(len(e['targets']) for e in trace['trace']) != 84):
        raise ValueError('requires the complete 84-target FP32 Q8 reference at context 512')
    report = dict(passed=False, pilot=args.pilot, model_path=trace['path'], context=512,
                  chunk_size=127, device=device_name(), torch=str(torch.__version__), graph=False,
                  storage='q8_0', activation_and_cache_dtype='float32',
                  comparison=('A: FP32 matvec for rows <=4; B: blocked-reduction matvec for rows 1-4. '
                              'Both use compensated BF16 prefill; A is October arm A unchanged.'),
                  limitations=['pacing inserts idle between synchronous operations only; timed event '
                               'windows still bracket one synchronize/event/synchronize',
                               'internal decode and memory telemetry outstanding'],
                  passes=[], pacing={}, phases={})
    pacer = thermal_pacing.ThermalPacer(report['pacing'], soft_stop=args.soft_stop,
                                         poll_seconds=args.poll_seconds)
    deactivate = None
    try:
        synchronize()
        started = time.perf_counter()
        deactivate = thermal_pacing.activate(pacer)
        model = ModelWeights(trace['path'], 512)
        if model.quantization != 'q8_0':
            raise RuntimeError('the gate requires Q8 storage; set DAMENDAEMON_QUANTIZATION=q8_0')
        report['phases']['model_seconds'] = time.perf_counter()-started
        synchronize()
        started = time.perf_counter()
        executor = make_executor(model, use_graph=False, max_batch=4)
        report['phases']['executor_seconds'] = time.perf_counter()-started
        report['phases']['kernel_backend'] = getattr(executor, 'kernel_backend', None)
        quantized_kernels.linear = dispatch
        protocol = (('warmup-A', False, False), ('A1', False, True)) if args.pilot else (
                   ('warmup-A', False, False), ('warmup-B', True, False),
                   ('A1', False, True), ('B1', True, True), ('B2', True, True), ('A2', False, True))
        for name, arm, measured in protocol:
            blocked_arm = arm
            synchronize()
            started = time.perf_counter()
            result = replay(executor, trace['trace'], pacer)
            result.update(name=name, measured=measured, blocked_arm=arm)
            report['passes'].append(result)
            report['phases'][f'{name}_seconds'] = time.perf_counter()-started
            if not result['passed']:
                raise AssertionError(f'full numerical gate failed in {name}')
            if measured:
                result['decode_median_seconds'] = {
                    str(size): statistics.median(row['seconds'] for row in result['timings']
                        if row['kind'] == 'decode' and len(row['slots']) == size)
                    for size in (1, 2, 4)}
            print(f'{name}: 84/84 passed', flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        if deactivate is not None:
            deactivate()
        quantized_kernels.linear = original_linear
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
