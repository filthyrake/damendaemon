"""Replay independent teacher-forced logits in the production runtime environment.

Generate the local trace with verify_stress.py --golden-output. Only trusted,
locally generated traces are accepted; torch.load uses weights_only=True.
"""
import argparse
import json
import time
import torch
from dllm.qwen2 import ModelWeights
from dllm.runtime import require_gpu, make_executor
from dllm.device import device_name, synchronize


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace')
    p.add_argument('--no-graph', action='store_true')
    p.add_argument('--prefill-chunk-size',type=int,default=0)
    p.add_argument('--slot', type=int, help='replay one original slot in isolation to reduce memory use')
    p.add_argument('--model-path', help='local copy of the exact checkpoint used to generate the trace')
    args = p.parse_args()
    if args.prefill_chunk_size<0:raise ValueError('invalid chunk size')
    require_gpu()
    torch.set_num_threads(4)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    events = trace['trace']
    if args.slot is not None:
        selected = []
        for event in events:
            rows = [i for i, slot in enumerate(event['slots']) if slot == args.slot]
            if rows:
                selected.append({**event, 'slots': [0]*len(rows),
                    **{key: [event[key][i] for i in rows] for key in ('targets', 'prompts', 'tokens') if key in event}})
        if not selected:
            raise ValueError('selected slot is absent from the reference trace')
        events = selected
    capacity = max(max(event['slots']) for event in events) + 1
    model = ModelWeights(args.model_path or trace['path'], trace['context'])
    executor = make_executor(model, use_graph=not args.no_graph, max_batch=capacity)
    checks, timings = [], []
    for event in events:
        synchronize()
        start = time.perf_counter()
        if event['kind'] == 'prefill':
            if args.prefill_chunk_size:
                outputs=[]
                for prompt,slot in zip(event['prompts'],event['slots']):
                    for chunk_start in range(0,len(prompt),args.prefill_chunk_size):
                        logits=executor.prefill_chunk(prompt[chunk_start:chunk_start+args.prefill_chunk_size],slot,chunk_start,
                            final=chunk_start+args.prefill_chunk_size>=len(prompt))
                    outputs.append(logits)
                actual=torch.cat(outputs)
            else:
                actual = executor.prefill_batch(event['prompts'], event['slots'])
        else:
            actual, _ = executor.decode_batch(event['tokens'], event['slots'])
        synchronize()
        timings.append({'kind': event['kind'], 'seconds': time.perf_counter()-start,
                        'slots': event['slots'], 'lengths': [len(p) for p in event.get('prompts', [])]})
        for row, target in enumerate(event['targets']):
            value = actual[row].float().cpu()
            target = target.squeeze(0)
            error = value-target
            relative = (error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1.e-6)).item()
            absolute = error.abs().max().item()
            match = value.argmax().item() == target.argmax().item()
            checks.append({'kind': event['kind'], 'slot': event['slots'][row], 'relative_rms': relative,
                           'max_absolute': absolute, 'greedy_match': match,
                           'passed': relative < .01 and absolute < .5 and match})
    passed = all(c['passed'] for c in checks)
    print(json.dumps({'passed': passed, 'model_path': trace['path'], 'context': trace['context'],
        'local_model_path': str(model.path), 'source_slot': args.slot,
        'prefill_chunk_size':args.prefill_chunk_size, 'device': device_name(), 'torch': str(torch.__version__), 'reference_dtype': trace['reference_dtype'],
        'reference_torch': trace['reference_torch'], 'reference_quantization': trace.get('reference_quantization','none'),
        'weight_quantization': model.quantization, 'host_weight_bytes': model.bytes, 'reference_attention': trace.get('reference_attention', 'sdpa'), 'graph_requested': not args.no_graph,
        'kernel_backend': executor.kernel_backend, 'compute_dtype': str(executor.compute_dtype),
        'qwen3_precision': getattr(executor, 'qwen3_precision', None),
        'dense_precision': getattr(executor, 'dense_precision', None), 'checks': checks, 'timings': timings}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
