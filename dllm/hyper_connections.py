"""Independent floating hyper-connection primitives; no model admission.

Reference equations: DeepSeek-V4-Flash inference/model.py and inference/kernel.py
at 60d8d70770c6776ff598c94bb586a859a38244f1 (MIT). Inputs/parameters must be finite.
Mixing parameters and accumulation use FP32; residual storage retains its dtype.
"""
import torch
from torch.nn import functional as F


def _positive(value, name):
    if type(value) not in (int, float) or not 0 < value <= torch.finfo(torch.float32).max:
        raise ValueError(f'{name} must be finite and positive')


def _hidden(hidden):
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 4
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not 1 <= hidden.shape[-2] <= 16 or hidden.shape[-1] < 1):
        raise ValueError('hyper-connections require [batch, sequence, 1..16 streams, width] floating state')
    return hidden.shape[-2], hidden.shape[-1]


def _parameter(tensor, shape, device):
    if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
            or tensor.dtype != torch.float32 or tensor.device != device):
        raise ValueError('hyper-connection parameters must have matching FP32 shape and device')


def mixing_weights(mixes, scale, bias, streams, *, iterations=20, epsilon=1e-6):
    """Split learned pre/post weights and normalize source/destination mixing."""
    if (type(streams) is not int or not 1 <= streams <= 16
            or type(iterations) is not int or not 1 <= iterations <= 128
            or not isinstance(mixes, torch.Tensor) or mixes.ndim != 3
            or mixes.dtype != torch.float32 or mixes.shape[-1] != streams*(streams+2)):
        raise ValueError('invalid hyper-connection mixing geometry or iteration count')
    _positive(epsilon, 'connection epsilon')
    _parameter(scale, (3,), mixes.device)
    _parameter(bias, (streams*(streams+2),), mixes.device)
    pre = (mixes[..., :streams]*scale[0] + bias[:streams]).sigmoid() + epsilon
    post = 2*(mixes[..., streams:2*streams]*scale[1] + bias[streams:2*streams]).sigmoid()
    matrix = (mixes[..., 2*streams:]*scale[2] + bias[2*streams:]).reshape(*mixes.shape[:2], streams, streams)
    matrix = matrix.softmax(-1) + epsilon
    matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
    for _ in range(iterations-1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + epsilon)
    return pre, post, matrix


def reduce_streams(hidden, projection, scale, bias, *, norm_epsilon=1e-6,
                   connection_epsilon=1e-6, iterations=20):
    """Produce a layer input and retained post/matrix coefficients."""
    streams, width = _hidden(hidden)
    _positive(norm_epsilon, 'normalization epsilon')
    _parameter(projection, (streams*(streams+2), streams*width), hidden.device)
    # Check the remaining static options before projection work starts.
    if type(iterations) is not int or not 1 <= iterations <= 128:
        raise ValueError('hyper-connection iteration count must be 1..128')
    _positive(connection_epsilon, 'connection epsilon')
    _parameter(scale, (3,), hidden.device)
    _parameter(bias, (streams*(streams+2),), hidden.device)
    state = hidden.float()
    flat = state.flatten(-2)
    mixes = F.linear(flat, projection) * (flat.square().mean(-1, keepdim=True)+norm_epsilon).rsqrt()
    pre, post, matrix = mixing_weights(mixes, scale, bias, streams,
                                      iterations=iterations, epsilon=connection_epsilon)
    reduced = torch.matmul(pre.unsqueeze(-2), state).squeeze(-2)
    return reduced.to(hidden.dtype), post, matrix


def restore_streams(output, residual, post, matrix):
    """Recombine streams with coefficients indexed as matrix[source, destination].

    Batched matrix multiplication avoids a [batch, sequence, streams, streams,
    width] broadcast allocation. Only the result has the residual state's size.
    """
    streams, width = _hidden(residual)
    batch, sequence = residual.shape[:2]
    if (not isinstance(output, torch.Tensor) or output.shape != (batch, sequence, width)
            or output.device != residual.device or output.dtype != residual.dtype):
        raise ValueError('hyper-connection output must match residual batch, sequence, width and dtype/device')
    _parameter(post, (batch, sequence, streams), residual.device)
    _parameter(matrix, (batch, sequence, streams, streams), residual.device)
    combined = torch.matmul(matrix.transpose(-1, -2), residual.float())
    return (combined + post.unsqueeze(-1)*output.float().unsqueeze(-2)).to(output.dtype)


def reduce_head(hidden, projection, scale, bias, *, norm_epsilon=1e-6, connection_epsilon=1e-6):
    """Reduce residual streams for the final normalization/output head."""
    streams, width = _hidden(hidden)
    _positive(norm_epsilon, 'normalization epsilon')
    _positive(connection_epsilon, 'connection epsilon')
    _parameter(projection, (streams, streams*width), hidden.device)
    _parameter(scale, (1,), hidden.device)
    _parameter(bias, (streams,), hidden.device)
    state = hidden.float()
    flat = state.flatten(-2)
    logits = F.linear(flat, projection) * (flat.square().mean(-1, keepdim=True)+norm_epsilon).rsqrt()
    coefficients = (logits*scale + bias).sigmoid()+connection_epsilon
    return torch.matmul(coefficients.unsqueeze(-2), state).squeeze(-2).to(hidden.dtype)
