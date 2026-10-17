"""Static rotary schedules, shared by CUDA and Metal execution.

Llama 3 preserves short wavelengths, scales long wavelengths, and smoothly
interpolates the band in between. Dynamic schedules need separate lifecycle
handling and are rejected by configuration validation.
"""
import math
import torch


def inverse_frequencies(config, device):
    dim = config['head_dim']
    frequencies = config['rope_theta'] ** (torch.arange(0,dim,2,device=device,dtype=torch.float32)/dim)
    inv = 1.0 / frequencies
    scaling = config.get('rope_scaling')
    if scaling is None:
        return inv
    factor = scaling['factor']
    if scaling['rope_type'] == 'linear':
        return inv/factor
    if scaling['rope_type'] == 'yarn':
        original = scaling.get('original_max_position_embeddings', config['max_position_embeddings'])
        if 'original_max_position_embeddings' in scaling:
            factor = config['max_position_embeddings'] / original
        def boundary(rotations):
            return dim * math.log(original / (2 * math.pi * rotations)) / (2 * math.log(config['rope_theta']))
        low = boundary(scaling.get('beta_fast', 32))
        high = boundary(scaling.get('beta_slow', 1))
        if scaling.get('truncate', True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(0, low), min(dim - 1, high)
        upper = high + .001 if low == high else high
        blend = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (upper - low)).clamp(0, 1)
        # Compute interpolation directly to match the specified frequency schedule.
        interpolated = 1.0 / (factor * frequencies)
        return interpolated * blend + inv * (1 - blend)
    wavelength = 2*math.pi/inv
    original = scaling['original_max_position_embeddings']
    low, high = scaling['low_freq_factor'], scaling['high_freq_factor']
    blend = (original/wavelength-low)/(high-low)
    middle = (1-blend)*inv/factor + blend*inv
    return torch.where(wavelength < original/high, inv,
                       torch.where(wavelength > original/low, inv/factor, middle))


def attention_factor(config):
    """Static scaling of both sine and cosine tables, including YaRN's mscale."""
    scaling = config.get('rope_scaling')
    if not scaling or scaling['rope_type'] != 'yarn':
        return 1.0
    if 'attention_factor' in scaling:
        return scaling['attention_factor']
    factor = scaling['factor']
    if 'original_max_position_embeddings' in scaling:
        factor = config['max_position_embeddings'] / scaling['original_max_position_embeddings']
    return 1 + .1 * math.log(factor)
