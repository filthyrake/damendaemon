"""Floating expert block foundations; no packed expert or model qualification.

Independent implementation of the activation/weighting order documented in
DeepSeek-V4-Flash inference/model.py at 60d8d70770c6776ff598c94bb586a859a38244f1
(MIT). The caller owns placement and memory admission; weights stay in their
provided floating representation rather than silently promoting whole matrices.
"""
import torch
from torch.nn import functional as F


def expert_feed_forward(hidden, gate_weight, up_weight, down_weight, *,
                        route_weights=None, activation_limit=None):
    """Compute one shared or routed SiLU expert on already resident tensors.

    Unweighted output is a shared-expert contribution. For routed output, supply
    one FP32 coefficient per row; weighting precedes down-projection input cast.
    This does not select/gather experts or implement FP4/FP8 checkpoint execution.
    """
    tensors = (hidden, gate_weight, up_weight, down_weight)
    if (any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in tensors)
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or any(t.dtype != hidden.dtype or t.device != hidden.device for t in tensors)):
        raise ValueError('floating expert tensors must be matrices with matching FP16/BF16/FP32 dtype and device')
    rows, width = hidden.shape
    intermediate = gate_weight.shape[0]
    if (width < 1 or intermediate < 1 or gate_weight.shape != (intermediate, width)
            or up_weight.shape != gate_weight.shape or down_weight.shape != (width, intermediate)):
        raise ValueError('expert projection shapes do not match hidden and intermediate widths')
    if (activation_limit is not None and (type(activation_limit) not in (int, float)
                                         or not 0 < activation_limit <= torch.finfo(torch.float32).max)):
        raise ValueError('expert activation limit must be finite and positive')
    if route_weights is not None and (not isinstance(route_weights, torch.Tensor)
            or route_weights.shape != (rows, 1) or route_weights.dtype != torch.float32
            or route_weights.device != hidden.device):
        raise ValueError('route weights must be one FP32 coefficient per row on the expert device')
    gate = F.linear(hidden, gate_weight).float()
    up = F.linear(hidden, up_weight).float()
    activated = activate_expert(gate, up, route_weights, activation_limit)
    return F.linear(activated.to(hidden.dtype), down_weight)


def activate_expert(gate, up, route_weights, activation_limit):
    """Internal FP32 activation order shared by individual and batched experts."""
    if activation_limit is not None:
        if not torch.isfinite(gate).all() or not torch.isfinite(up).all():
            raise ValueError('nonfinite expert projection before clipping')
        gate = gate.clamp(max=activation_limit)
        up = up.clamp(min=-activation_limit, max=activation_limit)
    activated = F.silu(gate) * up
    if route_weights is not None:
        activated = activated * route_weights
    return activated
