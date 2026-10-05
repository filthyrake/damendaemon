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


def weight_bytes(path,quantization=None):
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
    return ModelWeights(path,context,quantization=quantization)
