"""Explicit opt-in for the independently tested experimental image path."""
import os


def enabled():
    value=os.environ.get('DAMENDAEMON_EXPERIMENTAL_VISION','0')
    if value not in ('0','1'):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_VISION must be 0 or 1')
    return value=='1'


def eligible(config):
    text=config.get('model_type')=='qwen3_5_text'
    # Qualified text geometries: 0.8B and 2B (24-layer, 1024/2048-wide,
    # #573/#575) and Qwen3.8-27B (64-layer, 5120-wide; #50 A6000
    # encoder/full-model/service cells, 2026-10-31).
    return text and (config.get('num_hidden_layers'),config.get('hidden_size')) in ((24,1024),(24,2048),(64,5120))


def validate_model(config, context):
    if not eligible(config) or context > 1024:
        raise ValueError('image execution currently requires the qualified 24-layer 1024/2048-wide or 64-layer 5120-wide Qwen3.5 text geometry and context <=1024')


def validate_execution(model,max_batch):
    import torch
    from .device import selected_device, selected_accelerator
    device=selected_device()
    if selected_accelerator()=='rocm':
        # AMD Radeon 8060S (HOMER gfx1151) admitted on merit (#50, 2026-10-25):
        # encoder 75/75 checks at worst rel RMS 1.85e-05 against the pinned
        # transformers-5.18.0 CPU FP32 reference, full-model image gate 54/54
        # (worst RMS 4.1e-03 vs the 0.01 gate) with a healthy changed-image
        # control. Same gates, thresholds and geometry limits as the other
        # devices; unsupported ROCm devices keep the explicit error below.
        supported=(torch.cuda.is_available()
                   and torch.cuda.get_device_name()=='Radeon 8060S Graphics')
    else:
        # NVIDIA RTX A6000 admitted on merit (#50, 2026-10-31): encoder
        # 150/150 checks at worst rel RMS 2.04e-05 against the pinned
        # transformers-5.18.0 CPU FP32 reference, same depth-24-carried
        # gates; full-model image and HTTP service gates recorded in
        # docs/evidence/vision/qwen38-27b-a6000/. Same context/slot limits
        # and opt-in as every other device.
        supported=(torch.backends.mps.is_available() if device=='mps'
                  else torch.cuda.is_available() and torch.cuda.get_device_name() in ('NVIDIA A2','NVIDIA A30','NVIDIA RTX A6000'))
    if not supported or not 1<=max_batch<=4 or model.max_context>1024:
        raise ValueError('experimental vision serving requires NVIDIA A2, NVIDIA A30, NVIDIA RTX A6000, AMD Radeon 8060S (HOMER) or Apple MPS, context <=1024, and at most four slots')
