"""Explicit, bounded precision experiments for the ROCm hybrid text executor."""
import os

from .device import selected_accelerator


def select(model, use_graph, max_batch, kernels):
    mode = os.environ.get('DLLM_HYBRID_PRECISION', 'float16')
    if mode not in ('float16', 'float32', 'compensated'):
        raise ValueError('DLLM_HYBRID_PRECISION must be float16, float32 or compensated')
    if mode == 'float16':
        return mode
    if (selected_accelerator() != 'rocm' or model.quantization != 'q8_0'
            or model.config['model_type'] != 'qwen3_5_text'
            or model.max_context > 512 or max_batch > 4 or use_graph
            or kernels == 'torch' or os.environ.get('DLLM_HYBRID_PREFILL', 'scan') != 'scan'
            or getattr(model, 'vision_weights', None) is not None):
        raise ValueError('experimental hybrid precision requires ROCm Q8 text, context <=512, '
                         'at most four slots, Triton scan and --no-graph')
    return mode
