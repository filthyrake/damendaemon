"""Explicit, unqualified hybrid Q4 bring-up; separate from dense Q4 support."""
import os

from .device import device_name, selected_accelerator
from .vision_policy import enabled as vision_enabled


def _pipeline_pair_anchor():
    """True when a quantized two-device pipeline is explicitly opted in.

    #536 gate 2: under DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE=1 with the
    quant opt-in, stages build on each guarded pipeline device inside
    `torch.cuda.device(stage_device)`, so the single-device A30 identity pin
    cannot be evaluated with the current-device name. The anchor moves to the
    PRIMARY pipeline device (cuda:0) while the pipeline policy owns the pair,
    budgets, precision and context admission.
    """
    if selected_accelerator() != 'cuda':
        return False
    from .hybrid_pipeline_policy import budgets, quant_pipeline_enabled
    limits = budgets()
    if limits is None or len(limits) < 2 or not quant_pipeline_enabled():
        return False
    import torch
    return torch.cuda.get_device_name(0) == 'NVIDIA A30'


def validate(config, max_context=None):
    accelerator = selected_accelerator()
    if (os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4', '0') != '1'
            or accelerator not in ('rocm', 'cuda')
            or config.get('model_type') != 'qwen3_5_text'
            or os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'compensated'
            or (max_context is not None and not 1 <= max_context <= 512)
            or vision_enabled()
            or (accelerator == 'cuda' and device_name() != 'NVIDIA A30'
                and not _pipeline_pair_anchor())):
        raise ValueError('experimental hybrid Q4 requires explicit opt-in, ROCm or NVIDIA A30, '
                         'compensated text execution and context <=512')
