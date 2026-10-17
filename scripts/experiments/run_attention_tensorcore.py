"""Run an existing validation script with explicitly isolated attention dispatch."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys

import torch
from damendaemon import kernels
from scripts.experiments.attention_tensorcore import attention
from scripts.experiments.attention_tiles import attention as split_attention


def install_validation_dispatch(block, kind='tensorcore'):
    if block not in (64, 128):
        raise ValueError('unsupported experiment block')
    if kind not in ('tensorcore', 'split'):
        raise ValueError('unsupported attention experiment')
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise RuntimeError('attention experiment is limited to A2')
    original = kernels.attention
    signature = inspect.signature(original)
    source = 'attention_tensorcore.py' if kind == 'tensorcore' else 'attention_tiles.py'
    operation = attention if kind == 'tensorcore' else split_attention
    report = dict(kind=f'experimental fixed-cache {kind} attention',
                  block=block, captured_or_eager_calls=0, fallback_calls=0,
                  source_sha256=hashlib.sha256(Path(__file__).with_name(source).read_bytes()).hexdigest())

    def dispatch(*values, **options):
        bound = signature.bind(*values, **options)
        bound.apply_defaults()
        p = bound.arguments
        q, k, v = p['query'], p['cache_k'], p['cache_v']
        supported = (q.is_cuda and q.dtype == k.dtype == v.dtype == torch.float16
            and q.shape[1:] == (32, 1, 128) and 1 <= q.shape[0] <= 4
            and k.shape[1] == 8 and k.shape[2] <= 2048 and k.shape[3] == 128
            and all(t.is_contiguous() for t in (q, k, v))
            and not p['window'] and p['scale'] in (None, 128**-.5)
            and all(p[name] is None for name in ('alibi', 'page_table', 'relative_bias', 'sinks', 'scratch_bytes')))
        if not supported:
            report['fallback_calls'] += 1
            return original(*values, **options)
        report['captured_or_eager_calls'] += 1
        slots = p['slots']
        if slots is None:
            slots = torch.zeros(q.shape[0], device=q.device, dtype=torch.long)
        return operation(q, k, v, p['positions'], slots, block)

    kernels.attention = dispatch
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--block', type=int, choices=(64, 128), default=64)
    parser.add_argument('--kind', choices=('tensorcore', 'split'), default='tensorcore')
    parser.add_argument('script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original = kernels.attention
    report = install_validation_dispatch(args.block, args.kind)
    report.update(script=args.script, arguments=args.arguments, completed=False)
    saved_argv = sys.argv
    try:
        sys.argv = [args.script, *args.arguments]
        runpy.run_path(args.script, run_name='__main__')
        if not report['captured_or_eager_calls']:
            raise AssertionError('experiment was never selected')
        report['completed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        kernels.attention = original
        sys.argv = saved_argv
        Path(args.report).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
