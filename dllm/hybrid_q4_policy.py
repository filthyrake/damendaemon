"""Explicit, unqualified hybrid Q4 bring-up; separate from dense Q4 support."""
import os

from .device import device_name, selected_accelerator
from .vision_policy import enabled as vision_enabled


def validate(config, max_context=None):
    accelerator = selected_accelerator()
    if (os.environ.get('DLLM_EXPERIMENTAL_HYBRID_Q4', '0') != '1'
            or accelerator not in ('rocm', 'cuda')
            or config.get('model_type') != 'qwen3_5_text'
            or os.environ.get('DLLM_HYBRID_PRECISION') != 'compensated'
            or (max_context is not None and not 1 <= max_context <= 512)
            or vision_enabled()
            or (accelerator == 'cuda' and device_name() != 'NVIDIA A30')):
        raise ValueError('experimental hybrid Q4 requires explicit opt-in, ROCm or NVIDIA A30, '
                         'compensated text execution and context <=512')
