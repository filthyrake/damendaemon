"""Shared architecture dispatch for residency and registration."""
import json
from pathlib import Path


def config(path):
    value = json.loads((Path(path)/'config.json').read_text())
    if value.get('model_type')=='t5':
        from .t5 import config as t5_config
        return t5_config(path)
    if value.get('model_type') == 'bert':
        from .encoder import config as encoder_config
        return encoder_config(path)
    from .config import decoder_config
    return decoder_config(value)


def weight_bytes(path,quantization=None):
    if config(path).get('model_type')=='t5':
        from .t5 import weight_bytes as estimate
    elif config(path).get('task') == 'embedding':
        from .encoder import weight_bytes as estimate
    else:
        from .qwen2 import weight_bytes as estimate
    return estimate(path,quantization=quantization)


def load(path,context,quantization=None):
    if config(path).get('model_type')=='t5':
        from .t5 import T5Weights
        return T5Weights(path,context,quantization=quantization)
    if config(path).get('task') == 'embedding':
        from .encoder import EncoderWeights
        return EncoderWeights(path,context,quantization=quantization)
    from .qwen2 import ModelWeights
    return ModelWeights(path,context,quantization=quantization)
