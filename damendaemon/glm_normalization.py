"""Checked FP32 GLM RMS normalization for residual blocks and output heads."""
import torch


def rms_norm(value, weight, epsilon):
    variance=value.square().mean(-1,keepdim=True)
    # Finite inputs can overflow here and otherwise silently normalize to zero.
    if not torch.isfinite(variance).all():
        raise RuntimeError('nonfinite GLM normalization variance')
    result=value*torch.rsqrt(variance+epsilon)*weight
    if not torch.isfinite(result).all():
        raise RuntimeError('nonfinite GLM normalization output')
    return result
