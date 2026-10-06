"""Explicit, unqualified hybrid Q4 bring-up; separate from dense Q4 support."""
import os

from .device import selected_accelerator
from .vision_policy import enabled as vision_enabled


def validate(config, max_context=None):
    if (os.environ.get('DLLM_EXPERIMENTAL_HYBRID_Q4', '0') != '1'
            or selected_accelerator() != 'rocm'
            or config.get('model_type') != 'qwen3_5_text'
            or os.environ.get('DLLM_HYBRID_PRECISION') != 'compensated'
            or (max_context is not None and not 1 <= max_context <= 512)
            or vision_enabled()):
        raise ValueError('experimental hybrid Q4 requires explicit opt-in, ROCm, '
                         'compensated text execution and context <=512')
