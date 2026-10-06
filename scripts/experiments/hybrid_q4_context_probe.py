"""Isolated A30 Q4 context experiment for issue #70; production stays at 512."""
from contextlib import contextmanager
from copy import copy
import runpy
import sys

from dllm import hybrid_precision, hybrid_q4_policy
from dllm.device import selected_accelerator


@contextmanager
def context_probe():
    original_precision = hybrid_precision.select
    original_q4 = hybrid_q4_policy.validate

    def validate(config, max_context=None):
        if selected_accelerator() != 'cuda':
            raise ValueError('Q4 context probe requires the admitted A30 CUDA path')
        if max_context is not None and not 512 <= max_context <= 32768:
            raise ValueError('Q4 context probe is bounded to 512..32768 tokens')
        # The original policy retains A30 identity, explicit opt-in, precision,
        # architecture and image guards. Only context admission is substituted.
        return original_q4(config, None if max_context is None else 512)

    def select(model, use_graph, max_batch, kernels):
        if model.quantization != 'q4_0':
            raise ValueError('Q4 context probe cannot change another weight format')
        validate(model.config, model.max_context)
        admission = copy(model)
        admission.max_context = 512
        return original_precision(admission, use_graph, max_batch, kernels)

    hybrid_q4_policy.validate = validate
    hybrid_precision.select = select
    try:
        yield
    finally:
        hybrid_precision.select = original_precision
        hybrid_q4_policy.validate = original_q4


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('replay', 'serve'):
        raise SystemExit('choose replay TRACE --no-graph or serve [server options]')
    action = sys.argv.pop(1)
    print('UNQUALIFIED A30 Q4 context probe: production admission is unchanged; issue #70.', file=sys.stderr)
    with context_probe():
        if action == 'replay':
            runpy.run_path('scripts/replay_reference.py', run_name='__main__')
        else:
            sys.argv.insert(1, 'serve')
            runpy.run_module('dllm', run_name='__main__')


if __name__ == '__main__':
    main()
