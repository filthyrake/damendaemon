"""Isolated A6000 Q4 context experiment (#68 re-scope); production stays at 512.

Mirrors scripts/experiments/hybrid_q4_context_probe.py (the #68/#70 A30/HOMER
probe) with the device-identity gate substituted: this probe admits exactly
the desktop RTX A6000 by explicit UUID pin (stricter than the A30 probe's
device-name check — a UUID is unique to the card) behind the single-device
shared-CUDA guard. Every other guard from the production policy is
reimplemented verbatim: explicit DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4 opt-in,
qwen3_5_text architecture only, compensated precision, no vision, and the
context bound 512..32768. Production admission (hybrid_q4_policy.validate)
is NOT modified; only this process's isolated experiment view is.
"""
from contextlib import contextmanager
from copy import copy
import runpy
import sys

from damendaemon import hybrid_precision, hybrid_q4_policy
from damendaemon.device import device_name, selected_accelerator
from damendaemon.vision_policy import enabled as vision_enabled

A6000_UUID = 'GPU-4adb9ddb-7bb3-189c-0fa2-51d67d1189ed'


@contextmanager
def a6000_context_probe():
    original_precision = hybrid_precision.select
    original_q4 = hybrid_q4_policy.validate

    def validate(config, max_context=None):
        import os
        accelerator = selected_accelerator()
        if accelerator != 'cuda':
            raise ValueError('A6000 Q4 context probe requires the CUDA device')
        if device_name() != 'NVIDIA RTX A6000':
            raise ValueError('A6000 Q4 context probe is limited to the RTX A6000')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != A6000_UUID:
            raise ValueError('A6000 Q4 context probe requires the pinned '
                             'single-device guard UUID')
        # Guards reimplemented verbatim from hybrid_q4_policy.validate.
        if os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4', '0') != '1':
            raise ValueError('experimental hybrid Q4 requires explicit opt-in')
        if config.get('model_type') != 'qwen3_5_text':
            raise ValueError('Q4 context probe is qwen3_5_text only')
        if os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'compensated':
            raise ValueError('compensated text execution is required')
        if max_context is not None and not 512 <= max_context <= 32768:
            raise ValueError('Q4 context probe is bounded to 512..32768 tokens')
        if vision_enabled():
            raise ValueError('Q4 context probe is text-only (vision disabled)')

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
    print('UNQUALIFIED A6000 Q4 context probe (issue #68 re-scope): production admission is unchanged.', file=sys.stderr)
    with a6000_context_probe():
        if action == 'replay':
            runpy.run_path('scripts/replay_reference.py', run_name='__main__')
        else:
            sys.argv.insert(1, 'serve')
            runpy.run_module('damendaemon', run_name='__main__')


if __name__ == '__main__':
    main()
