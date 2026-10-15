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


def validate_model(config, context):
    if not eligible(config) or context > 1024:
        raise ValueError('image execution currently requires the qualified 1024-wide Qwen3.5 text geometry and context <=1024')


def validate_execution(model,max_batch):
    import torch
    from .device import selected_device
    device=selected_device()
    supported=(torch.backends.mps.is_available() if device=='mps'
               else torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A2')
    if not supported or not 1<=max_batch<=4 or model.max_context>1024:
        raise ValueError('experimental vision serving requires NVIDIA A2 or Apple MPS, context <=1024, and at most four slots')
