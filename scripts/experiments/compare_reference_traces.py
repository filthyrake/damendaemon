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
    p.add_argument('--require-pass', action='store_true',
                   help='require complete matching histories and the unchanged numerical gate')
    args = p.parse_args()
    left = torch.load(args.left, map_location='cpu', weights_only=True)
    right = torch.load(args.right, map_location='cpu', weights_only=True)
    if args.require_pass and (not left.get('complete', True) or not right.get('complete', True)):
        raise ValueError('a reference comparison gate requires completed traces')
    if left['path'] != right['path'] or left['context'] != right['context']:
        raise ValueError('checkpoint paths and configured contexts must match')
    if args.require_pass and left.get('reference_quantization') != right.get('reference_quantization'):
        raise ValueError('reference comparison gate requires identical weight quantization')
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
    passed = bool(checks) and not unmatched and all(check['relative_rms'] < .01
        and check['max_absolute'] < .5 and check['greedy_match'] for check in checks)
    print(json.dumps(dict(left=args.left, right=args.right, left_dtype=left['reference_dtype'],
                         right_dtype=right['reference_dtype'], checks=checks, unmatched_contexts=unmatched,
                         passed=passed if args.require_pass else None,
                         limitation='Reference-to-reference precision comparison, not runtime correctness or performance.'), indent=2))
    if args.require_pass and not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
