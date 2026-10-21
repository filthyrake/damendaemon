"""Explicit, bounded precision experiments for the hybrid text executor."""
import os

from .device import selected_accelerator


def select(model, use_graph, max_batch, kernels):
    if selected_accelerator() == 'xpu':
        from .xpu_policy import validate_model
        validate_model(model.config, model.quantization, model.max_context, max_batch)
        if (os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float32') != 'float32'
                or kernels not in ('auto', 'torch') or use_graph
                or os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan') != 'scan'
                or getattr(model, 'vision_weights', None) is not None):
            raise ValueError('Intel hybrid bring-up requires eager FP32 text without graphs or FLA')
        return 'float32'
    mode = os.environ.get('DAMENDAEMON_HYBRID_PRECISION', 'float16')
    if mode not in ('float16', 'float32', 'compensated'):
        raise ValueError('DAMENDAEMON_HYBRID_PRECISION must be float16, float32 or compensated')
    if model.quantization == 'q4_0':
        from .hybrid_q4_policy import validate
        validate(model.config, model.max_context)
    if mode == 'float16':
        return mode
    # Q4's policy independently admits the bounded A30 experiment; Q8 and
    # other nondefault precision modes retain their ROCm-only scope. The
    # explicit distributed-chain opt-in (#536 gate 4) admits the chain's
    # qualified context cell for Q4; every other bound still applies.
    from .hybrid_q4_policy import CHAIN_CONTEXT_LIMIT, chain_anchor
    context_bound = CHAIN_CONTEXT_LIMIT if (chain_anchor()
                                            and model.quantization == 'q4_0') else 512
    supported_device = selected_accelerator() == 'rocm' or model.quantization == 'q4_0'
    if (not supported_device or model.quantization not in ('q8_0', 'q4_0')
            or model.config['model_type'] != 'qwen3_5_text'
            or model.max_context > context_bound or max_batch > 4 or use_graph
            or kernels == 'torch' or os.environ.get('DAMENDAEMON_HYBRID_PREFILL', 'scan') != 'scan'
            or getattr(model, 'vision_weights', None) is not None):
        raise ValueError('experimental hybrid precision requires an admitted packed-text device, '
                         f'context <={context_bound}, at most four slots, Triton scan and --no-graph')
    return mode
