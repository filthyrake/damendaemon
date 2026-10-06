"""Shared architecture dispatch for residency and registration."""
import json
from pathlib import Path


def is_gguf(path):
    """HF cache symlinks resolve to extensionless blobs: inspect file magic too."""
    path = Path(path)
    if not path.is_file():
        return False
    if path.suffix.lower() == '.gguf':
        return True
    with path.open('rb') as file:
        return file.read(4) == b'GGUF'


def config(path):
    if is_gguf(path):
        from .gguf_model import plan
        return dict(plan(path).config)
    value = json.loads((Path(path)/'config.json').read_text())
    quantization = value.get('quantization_config')
    if isinstance(quantization, dict) and quantization.get('quant_method') == 'awq':
        from .awq_model import config as awq_config
        return awq_config(path)
    if value.get('model_type')=='mamba2':
        from .mamba2 import config as mamba_config
        return mamba_config(path)
    if value.get('model_type')=='mamba':
        from .mamba import config as mamba_config
        return mamba_config(path)
    if value.get('model_type')=='t5':
        from .t5 import config as t5_config
        return t5_config(path)
    if value.get('model_type') in ('bert','xlm-roberta'):
        from .encoder import config as encoder_config
        return encoder_config(path)
    from .config import decoder_config
    return decoder_config(value)


def validate_device_model(path, quantization, context=None):
    from .device import selected_device
    from .layer_stream import enabled as offload_enabled, validate as validate_offload
    if offload_enabled():
        from .quantization import mode
        validate_offload(config(path), mode(quantization), 1 if context is None else context)
    if selected_device() != 'xpu':
        return
    if is_gguf(path):
        raise ValueError('Intel XPU does not support GGUF import')
    from .quantization import mode
    from .xpu_policy import validate_model
    validate_model(config(path), mode(quantization), context)


def weight_bytes(path,quantization=None):
    validate_device_model(path, quantization)
    if is_gguf(path):
        from .gguf_model import weight_bytes as estimate
        return estimate(path, quantization)
    if config(path).get('weight_format') == 'awq_gemm':
        from .awq_model import weight_bytes as estimate
        return estimate(path, quantization)
    if config(path).get('model_type')=='mamba2':
        from .mamba2 import weight_bytes as estimate
    elif config(path).get('model_type')=='mamba':
        from .mamba import weight_bytes as estimate
    elif config(path).get('model_type')=='t5':
        from .t5 import weight_bytes as estimate
    elif config(path).get('task') in ('embedding','rerank'):
        from .encoder import weight_bytes as estimate
    else:
        from .qwen2 import weight_bytes as estimate
    return estimate(path,quantization=quantization)


def load(path,context,quantization=None):
    validate_device_model(path, quantization, context)
    if is_gguf(path):
        from .gguf_model import GgufWeights
        return GgufWeights(path, context, quantization)
    if config(path).get('weight_format') == 'awq_gemm':
        from .awq_model import AwqWeights
        return AwqWeights(path, context, quantization)
    if config(path).get('model_type')=='mamba2':
        from .mamba2 import Mamba2Weights
        return Mamba2Weights(path,context,quantization=quantization)
    if config(path).get('model_type')=='mamba':
        from .mamba import MambaWeights
        return MambaWeights(path,context,quantization=quantization)
    if config(path).get('model_type')=='t5':
        from .t5 import T5Weights
        return T5Weights(path,context,quantization=quantization)
    if config(path).get('task') in ('embedding','rerank'):
        from .encoder import EncoderWeights
        return EncoderWeights(path,context,quantization=quantization)
    from .qwen2 import ModelWeights
    from .vision_policy import enabled,eligible
    if enabled() and eligible(config(path)):
        from .vision_model import VisionModelWeights
        from .vision_qwen35 import VisionWeights
        return VisionModelWeights(ModelWeights(path,context,quantization=quantization),VisionWeights(path))
    return ModelWeights(path,context,quantization=quantization)
