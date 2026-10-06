"""Independent FP32 reference for a bounded slice of a trusted local trace."""
import argparse
import json
from pathlib import Path
import sys
import torch
from reference_model import load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--slot', type=int, required=True)
    p.add_argument('--last-event', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.threads < 1 or args.last_event < 0:
        p.error('positive threads and nonnegative last event required')
    torch.set_num_threads(args.threads)
    source = torch.load(args.trace, map_location='cpu', weights_only=True)
    selected = []
    for index, event in enumerate(source['trace']):
        if index > args.last_event:
            break
        if args.slot not in event['slots']:
            continue
        row = event['slots'].index(args.slot)
        selected.append({**event, 'source_event_index': index, 'slots': [args.slot],
                         **{key: [event[key][row]] for key in ('targets', 'prompts', 'tokens') if key in event}})
    if not selected or selected[0]['kind'] != 'prefill':
        raise ValueError('selected slice must start with prefill')
    baseline = {**source, 'trace': selected}
    torch.save(baseline, args.output.with_suffix('.input.pt'))
    model = load_reference(source['path'], reference_quantization=source['reference_quantization'],
                           packed_cpu=True, local_files_only=True, trust_remote_code=False,
                           torch_dtype=torch.float32, attn_implementation='sdpa').eval()
    result = {**source, 'reference_dtype': 'float32', 'reference_torch': str(torch.__version__),
              'reference_attention': 'sdpa', 'packed_cpu_reference': True,
              'source_trace': str(args.trace), 'source_slot': args.slot,
              'source_last_event': args.last_event, 'trace': [], 'complete': False}
    cache = None
    with torch.inference_mode():
        for event in selected:
            print(f"reference source event {event['source_event_index']}: {event['kind']}", file=sys.stderr, flush=True)
            if event['kind'] == 'prefill':
                ids, cache = event['prompts'][0], None
            else:
                ids = [event['tokens'][0]]
            output = model(torch.tensor([ids]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            target = output.logits[:, -1].detach().float().cpu()
            if not torch.isfinite(target).all():
                raise ValueError('reference emitted non-finite logits')
            result['trace'].append({**event, 'targets': [target]})
            torch.save(result, args.output.with_suffix('.partial.pt'))
    result['complete'] = True
    torch.save(result, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'trace'}
                     | {'targets': len(result['trace']), 'threads': args.threads}, indent=2))


if __name__ == '__main__':
    main()
