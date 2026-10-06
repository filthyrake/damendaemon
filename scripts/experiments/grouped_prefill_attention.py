"""Independent #68 candidate: shared KV views instead of materialized GQA copies."""
import torch
from torch.nn import functional as F


def attention(q, k, v, **kwargs):
    kwargs.pop('enable_gqa', None)
    repeats = q.shape[1] // k.shape[1]
    outputs = []
    for head in range(k.shape[1]):
        query = q[:, head*repeats:(head+1)*repeats]
        key = k[:, head:head+1].expand(-1, repeats, -1, -1)
        value = v[:, head:head+1].expand(-1, repeats, -1, -1)
        outputs.append(F.scaled_dot_product_attention(query, key, value, **kwargs))
    return torch.cat(outputs, dim=1)


def matmul_attention(q, k, v):
    """Reuse each KV matrix across its flattened query heads with FP32 GEMMs."""
    batch, heads, length, dim = q.shape
    repeats = heads // k.shape[1]
    prefix = k.shape[2] - length
    mask = torch.arange(k.shape[2], device=q.device)[None, :] > (
        prefix + torch.arange(length, device=q.device)[:, None])
    outputs = []
    for head in range(k.shape[1]):
        query = q[:, head*repeats:(head+1)*repeats].reshape(batch, repeats*length, dim)
        scores = (query @ k[:, head].transpose(-1, -2)) * (dim**-.5)
        scores = scores.reshape(batch, repeats, length, k.shape[2])
        scores.masked_fill_(mask, -float('inf'))
        probabilities = scores.softmax(-1).reshape(batch, repeats*length, k.shape[2])
        outputs.append((probabilities @ v[:, head]).reshape(batch, repeats, length, dim))
    return torch.cat(outputs, dim=1)
