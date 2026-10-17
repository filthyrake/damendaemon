"""Fail closed before reference arithmetic can conceal a nonfinite intermediate."""
import torch


def finite(value, stage):
    if not torch.isfinite(value).all():
        raise ValueError(f'nonfinite reference {stage}')
    return value
