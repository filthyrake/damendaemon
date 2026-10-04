"""Static rotary schedules, shared by CUDA and Metal execution.

Llama 3 preserves short wavelengths, scales long wavelengths, and smoothly
interpolates the band in between. Dynamic schedules need separate lifecycle
handling and are rejected by configuration validation.
"""
import math
import torch


def inverse_frequencies(config, device):
    dim = config['head_dim']
    inv = 1.0 / (config['rope_theta'] **
                 (torch.arange(0,dim,2,device=device,dtype=torch.float32)/dim))
    scaling = config.get('rope_scaling')
    if scaling is None:
        return inv
    factor = scaling['factor']
    if scaling['rope_type'] == 'linear':
        return inv/factor
    wavelength = 2*math.pi/inv
    original = scaling['original_max_position_embeddings']
    low, high = scaling['low_freq_factor'], scaling['high_freq_factor']
    blend = (original/wavelength-low)/(high-low)
    middle = (1-blend)*inv/factor + blend*inv
    return torch.where(wavelength < original/high, inv,
                       torch.where(wavelength > original/low, inv/factor, middle))
