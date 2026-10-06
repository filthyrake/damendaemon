"""Compare trusted local reference traces only where full token histories match."""
import argparse
import json
import torch


def rows(trace):
    history = {}
    result = []
    for index, event in enumerate(trace['trace']):
        for row, slot in enumerate(event['slots']):
            if event['kind'] == 'prefill':
                history[slot] = list(event['prompts'][row])
            else:
                history[slot].append(event['tokens'][row])
            result.append((index, slot, tuple(history[slot]), event['targets'][row].float().flatten()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left')
    p.add_argument('right')
    args = p.parse_args()
    left = torch.load(args.left, map_location='cpu', weights_only=True)
    right = torch.load(args.right, map_location='cpu', weights_only=True)
    if left['path'] != right['path'] or left['context'] != right['context']:
        raise ValueError('checkpoint paths and configured contexts must match')
    lr, rr = rows(left), rows(right)
    if len(lr) != len(rr):
        raise ValueError('different target counts')
    checks, unmatched = [], []
    for a, b in zip(lr, rr):
        if a[:2] != b[:2]:
            raise ValueError('different event/slot ordering')
        index, slot = a[:2]
        if a[2] != b[2]:
            unmatched.append(dict(event_index=index, slot=slot))
            continue
        delta = a[3] - b[3]
        checks.append(dict(event_index=index, slot=slot, tokens=len(a[2]),
                           relative_rms=(delta.square().mean().sqrt()/b[3].square().mean().sqrt().clamp_min(1e-6)).item(),
                           max_absolute=delta.abs().max().item(), greedy_match=a[3].argmax().item()==b[3].argmax().item()))
    print(json.dumps(dict(left=args.left, right=args.right, left_dtype=left['reference_dtype'],
                         right_dtype=right['reference_dtype'], checks=checks, unmatched_contexts=unmatched,
                         limitation='Reference-to-reference precision comparison, not runtime correctness or performance.'), indent=2))


if __name__ == '__main__':
    main()
