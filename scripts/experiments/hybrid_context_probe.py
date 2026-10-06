"""Explicit unqualified context probe; does not change serving admission defaults.

Use ``python -m scripts.experiments.hybrid_context_probe replay TRACE --no-graph``
or ``... serve --model ... --no-graph``. All existing precision/device,
quantization, graph, batch and text-only guards remain in effect. Only this
process may probe compensated context above 512, up to 32768.
"""
from contextlib import contextmanager
from copy import copy
import runpy
import sys

from dllm import hybrid_precision


@contextmanager
def context_probe():
    original = hybrid_precision.select

    def select(model, use_graph, max_batch, kernels):
        if model.quantization != 'q8_0':
            raise ValueError('context probe is scoped to Q8; Q4 needs its own context gate')
        if not 512 <= model.max_context <= 32768:
            raise ValueError('context probe is bounded to 512..32768 tokens')
        admission = copy(model)
        admission.max_context = 512
        mode = original(admission, use_graph, max_batch, kernels)
        if mode != 'compensated':
            raise ValueError('context probe requires explicit compensated precision')
        return mode

    hybrid_precision.select = select
    try:
        yield
    finally:
        hybrid_precision.select = original


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('replay', 'serve'):
        raise SystemExit('choose replay TRACE --no-graph or serve [server options]')
    action = sys.argv.pop(1)
    print('UNQUALIFIED context probe: production admission is unchanged; issue #68.', file=sys.stderr)
    with context_probe():
        if action == 'replay':
            runpy.run_path('scripts/replay_reference.py', run_name='__main__')
        else:
            sys.argv.insert(1, 'serve')
            runpy.run_module('dllm', run_name='__main__')


if __name__ == '__main__':
    main()
