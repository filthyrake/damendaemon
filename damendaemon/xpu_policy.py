"""Opt-in Intel bring-up limits. Local tests do not qualify XPU execution."""
import os

DISPLAY_RESERVE_BYTES = 2 * 1024**3


def validate_environment():
    if os.environ.get('DAMENDAEMON_EXPERIMENTAL_XPU') != '1':
        raise ValueError('Intel XPU requires DAMENDAEMON_EXPERIMENTAL_XPU=1; execution is unqualified')
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:gpu':
        raise ValueError('Intel XPU requires ONEAPI_DEVICE_SELECTOR=level_zero:gpu')
    if os.environ.get('DAMENDAEMON_KERNELS', 'auto') not in ('auto', 'torch'):
        raise ValueError('Intel XPU currently requires eager torch kernels')
    for name in ('DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT', 'DAMENDAEMON_KV_CACHE_TOKENS',
                 'DAMENDAEMON_SPECULATIVE_TOKENS', 'DAMENDAEMON_PREFILL_GRAPHS', 'DAMENDAEMON_EXPERIMENTAL_VISION'):
        if os.environ.get(name, '0') != '0':
            raise ValueError(f'Intel XPU does not support {name}')


def validate_runtime(torch):
    validate_environment()
    xpu = getattr(torch, 'xpu', None)
    if xpu is None or not xpu.is_available():
        raise RuntimeError('Intel XPU is not available in this PyTorch environment')
    if xpu.device_count() != 1:
        raise RuntimeError('Intel bring-up requires exactly one visible XPU')
    if 'Intel' not in xpu.get_device_name(0) or 'B580' not in xpu.get_device_name(0):
        raise RuntimeError('Intel bring-up currently targets the Arc B580 only')
    xpu.set_device(0)


def validate_limits(context, batch=1):
    if type(context) is not int or not 1 <= context <= 2048:
        raise ValueError('Intel bring-up context must be 1..2048')
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError('Intel bring-up batch capacity must be 1..4')


def validate_model(config, quantization, context=None, batch=1):
    validate_environment()
    if (config.get('model_type') not in ('qwen3', 'qwen3_5_text') or quantization != 'none'
            or config.get('weight_format') or config.get('quantization_config')
            or config.get('rope_scaling')):
        raise ValueError('Intel bring-up requires Qwen3 dense or Qwen3.5-family hybrid text safetensors without quantization or scaled RoPE')
    validate_limits(1 if context is None else context, batch)
    if context is not None and context > config['max_position_embeddings']:
        raise ValueError('Intel bring-up context exceeds model positions')


def available_memory(torch):
    free, total = torch.xpu.mem_get_info()
    if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
        raise RuntimeError('invalid Intel XPU free-memory telemetry')
    return max(0, free - DISPLAY_RESERVE_BYTES)
