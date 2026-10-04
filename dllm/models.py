"""Shared architecture dispatch for residency and registration."""
import json
from pathlib import Path


def config(path):
    value = json.loads((Path(path)/'config.json').read_text())
    if value.get('model_type') == 'bert':
        from .encoder import config as encoder_config
        return encoder_config(path)
    from .config import decoder_config
    return decoder_config(value)


def weight_bytes(path):
    if config(path).get('task') == 'embedding':
        from .encoder import weight_bytes as estimate
    else:
        from .qwen2 import weight_bytes as estimate
    return estimate(path)


def load(path,context):
    if config(path).get('task') == 'embedding':
        from .encoder import EncoderWeights
        return EncoderWeights(path,context)
    from .qwen2 import ModelWeights
    return ModelWeights(path,context)
