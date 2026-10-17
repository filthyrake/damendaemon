"""Explicit same-device CUDA FP32 GLM stateless attention primitives.

These reuse CPU-reference equations and budgets, including validation syncs.
They are not fused kernels, a persistent cache, indexer or model qualification.
"""
import torch
from .glm_attention import _project_mla, _sparse_latent_attention


def _device(tensor):
    if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cuda':
        raise ValueError('explicit GLM CUDA entrypoint requires a CUDA tensor')
    return tensor.device


def project_mla(hidden,positions,weights,**options):
    return _project_mla(hidden,positions,weights,device=_device(hidden),**options)


def sparse_latent_attention(query,latents,rotary_keys,kv_weight,indices,positions,**options):
    return _sparse_latent_attention(query,latents,rotary_keys,kv_weight,indices,positions,
                                    device=_device(query),**options)
