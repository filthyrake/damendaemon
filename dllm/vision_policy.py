"""Explicit opt-in for the independently tested experimental image path."""
import os


def enabled():
    value=os.environ.get('DLLM_EXPERIMENTAL_VISION','0')
    if value not in ('0','1'):
        raise ValueError('DLLM_EXPERIMENTAL_VISION must be 0 or 1')
    return value=='1'


def eligible(config):
    return (config.get('model_type')=='qwen3_5_text' and config.get('hidden_size')==1024
            and config.get('num_hidden_layers')==24)
