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
    unsupported = {
        'DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT':
            'Intel XPU does not join the single-process layer split (a process '
            'cannot hold both vendors\' torch builds); an XPU rank participates '
            'in fleets through the host-staged two-process chain, qualified in '
            'docs/evidence/multi-gpu/cross-vendor-cell-600/ (#614)',
        'DAMENDAEMON_KV_CACHE_TOKENS': 'Intel XPU does not support paged KV cache',
        'DAMENDAEMON_SPECULATIVE_TOKENS': 'Intel XPU does not support speculative decoding',
        'DAMENDAEMON_PREFILL_GRAPHS': 'Intel XPU has no graph capture API in the '
                                      'qualified torch build (probed live, #605)',
        'DAMENDAEMON_EXPERIMENTAL_VISION': 'Intel XPU does not support vision execution',
    }
    for name, reason in unsupported.items():
        if os.environ.get(name, '0') != '0':
            raise ValueError(f'{name}: {reason}')


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


def validate_limits(context, batch=1, quantization='none'):
    limit = context_limit(quantization)
    if type(context) is not int or not 1 <= context <= limit:
        raise ValueError(f'Intel bring-up context must be 1..{limit} for {quantization!r} weights')
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError('Intel bring-up batch capacity must be 1..4')


def context_limit(quantization):
    # Dense Q8 weights leave headroom for the FP32 KV cache: the 0.6B dense
    # fixture needs ~0.22 MB/token, so 16K batch-1 fits beside the display
    # reserve. FP32 weights keep the original 2048 bring-up limit; Q4 stays
    # rejected entirely and hybrid Q8 is bounded by its geometry policy.
    return 16384 if quantization == 'q8_0' else 2048


def validate_model(config, quantization, context=None, batch=1):
    validate_environment()
    model_type = config.get('model_type')
    if quantization == 'q8_0':
        from .hybrid_metal import eligible as hybrid_eligible
        supported = model_type == 'qwen3' or hybrid_eligible(config, 1 if context is None else context, batch)
        if not supported:
            raise ValueError('Intel XPU Q8 supports dense Qwen3 and the qualified Qwen3.5-0.8B hybrid geometry only')
    if (model_type not in ('qwen3', 'qwen3_5_text') or quantization not in ('none', 'q8_0')
            or config.get('weight_format') or config.get('quantization_config')
            or config.get('rope_scaling')):
        raise ValueError('Intel bring-up requires Qwen3 dense or Qwen3.5-family hybrid text safetensors without quantization or scaled RoPE')
    validate_limits(1 if context is None else context, batch, quantization)
    if context is not None and context > config['max_position_embeddings']:
        raise ValueError('Intel bring-up context exceeds model positions')


def available_memory(torch):
    free, total = torch.xpu.mem_get_info()
    if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
        raise RuntimeError('invalid Intel XPU free-memory telemetry')
    return max(0, free - DISPLAY_RESERVE_BYTES)
