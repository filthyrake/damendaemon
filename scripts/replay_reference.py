"""Replay independent teacher-forced logits in the production runtime environment.

Generate the local trace with verify_stress.py --golden-output. Only trusted,
locally generated traces are accepted; torch.load uses weights_only=True.
"""
import argparse
import json
import os
import time
import torch
from dllm.qwen2 import ModelWeights
from dllm.runtime import require_gpu, make_executor
from dllm.device import device_name, selected_accelerator, synchronize
from dllm.quantization import mode
from scripts.reference_trace import validate_replay


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('trace')
    p.add_argument('--no-graph', action='store_true')
    p.add_argument('--cpu-threads', type=int, choices=range(1, 33), default=4,
                   help='bound CPU loading/quantization parallelism (default:4)')
    p.add_argument('--scalar',action='store_true',help='exercise the direct single-request decoder; requires --slot')
    p.add_argument('--prefill-chunk-size',type=int,default=0)
    p.add_argument('--slot', type=int, help='replay one original slot in isolation to reduce memory use')
    p.add_argument('--model-path', help='local copy of the exact checkpoint used to generate the trace')
    p.add_argument('--allow-quantization-mismatch', action='store_true',
                   help='explicit quality comparison; never a matched-format numerical qualification')
    args = p.parse_args()
    if args.scalar and args.slot is None:raise ValueError('--scalar requires --slot')
    if args.prefill_chunk_size<0:raise ValueError('invalid chunk size')
    require_gpu()
    torch.set_num_threads(args.cpu_threads)
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    validate_replay(trace, mode(), allow_quantization_mismatch=args.allow_quantization_mismatch)
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
    for event_index,event in enumerate(events):
        synchronize()
        start = time.perf_counter()
        if event['kind'] == 'prefill':
            if args.prefill_chunk_size:
                # This event replaces all listed slots. Retire their old pages
                # before replaying the replacement prompts sequentially.
                if hasattr(executor,'release_request'):
                    for slot in event['slots']:
                        executor.release_request(slot)
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
            actual, _ = (executor.decode(event['tokens'][0]) if args.scalar
                else executor.decode_batch(event['tokens'], event['slots']))
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
            checks.append({'event_index':event_index,'kind': event['kind'], 'slot': event['slots'][row], 'relative_rms': relative,
                           'max_absolute': absolute, 'greedy_match': match,
                           'passed': relative < .01 and absolute < .5 and match})
            if not checks[-1]['passed']:
                checks[-1].update(actual_top5=list(zip(value.topk(5).indices.tolist(),value.topk(5).values.tolist())),
                                  reference_top5=list(zip(target.topk(5).indices.tolist(),target.topk(5).values.tolist())))
    passed = all(c['passed'] for c in checks)
    print(json.dumps({'passed': passed, 'model_path': trace['path'], 'context': trace['context'],
        'quantization_matches_reference': model.quantization == trace.get('reference_quantization', 'none'),
        'allow_quantization_mismatch': args.allow_quantization_mismatch,
        'local_model_path': str(model.path), 'source_slot': args.slot, 'scalar':args.scalar,
        'prefill_chunk_size':args.prefill_chunk_size, 'cpu_threads':args.cpu_threads, 'device': device_name(), 'accelerator':selected_accelerator(), 'torch': str(torch.__version__), 'reference_dtype': trace['reference_dtype'],
        'rocm_aotriton_experimental':os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'),
        'reference_torch': trace['reference_torch'], 'packed_cpu_reference':trace.get('packed_cpu_reference',False), 'reference_quantization': trace.get('reference_quantization','none'),
        'reference_device': trace.get('reference_device'), 'packed_gpu_reference':trace.get('packed_gpu_reference',False),
        'weight_cache_status':model.weight_cache_status, 'weight_quantization': model.quantization, 'host_weight_bytes': model.bytes, 'reference_attention': trace.get('reference_attention', 'sdpa'), 'graph_requested': not args.no_graph,
        'kernel_backend': executor.kernel_backend,
        'compute_dtype': str(executor.compute_dtype) if hasattr(executor,'compute_dtype') else None,
        'hybrid_precision': getattr(executor, 'hybrid_precision', None),
        'qwen3_precision': getattr(executor, 'qwen3_precision', None),
        'olmoe_grouped_prefill': getattr(executor, 'grouped_prefill', None),
        'llama_precision': getattr(executor, 'llama_precision', None),
        'residual_dtype': str(executor.residual_dtype),
        'dense_precision': getattr(executor, 'dense_precision', None), 'checks': checks, 'timings': timings}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
