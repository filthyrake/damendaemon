"""Full-trace, warmed ABBA dispatch comparison; not an HTTP serving benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from scripts.experiments import replay_hybrid_bf16x3 as policy
from damendaemon.device import synchronize, device_name
from damendaemon.qwen2 import ModelWeights
from damendaemon.runtime import make_executor, require_gpu


@torch.inference_mode()
def replay(executor, events, minimum_rows):
    policy.minimum_rows = minimum_rows
    checks, timings = [], []
    for index, event in enumerate(events):
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
    return dict(minimum_compensated_rows=minimum_rows, checks=checks, timings=timings,
                passed=all(row['passed'] for row in checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require_gpu()
    torch.set_num_threads(4)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    if (not trace.get('complete', True) or trace['context'] != 512
            or trace['reference_dtype'] != 'float32' or trace['reference_quantization'] != 'q8_0'
            or len(trace['trace']) != 29 or sum(len(e['targets']) for e in trace['trace']) != 84):
        raise ValueError('requires the complete 84-target FP32 Q8 reference at context 512')
    report = dict(passed=False, model_path=trace['path'], context=512, chunk_size=127,
                  device=device_name(), torch=str(torch.__version__), graph=False,
                  storage='q8_0', activation_and_cache_dtype='float32',
                  comparison='A: FP32 matvec for rows <=4; B: compensated BF16 products for rows >=2. Both use compensated prefill.',
                  passes=[])
    try:
        model = ModelWeights(trace['path'], 512)
        executor = make_executor(model, use_graph=False, max_batch=4)
        # Each replay resets every slot via prompt replacement before decoding.
        for name, minimum, measured in (('warmup-A', 5, False), ('warmup-B', 2, False),
                                        ('A1', 5, True), ('B1', 2, True),
                                        ('B2', 2, True), ('A2', 5, True)):
            result = replay(executor, trace['trace'], minimum)
            result.update(name=name, measured=measured)
            report['passes'].append(result)
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
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
