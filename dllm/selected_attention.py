"""Chunked shared-KV attention equations for DeepSeek component development.

Independent implementation of the selected-KV and learned sink contract from
DeepSeek-V4-Flash inference/kernel.py (MIT), revision
60d8d70770c6776ff598c94bb586a859a38244f1. This uses FP32 arithmetic, not the
reference accelerator's intermediate probability rounding. No GPU qualification.
"""
import torch


@torch.no_grad()
def selected_attention(query, kv, indices, sink, *, scale=None, query_chunk=16, key_chunk=64):
    """Attend to per-query selected shared key/value vectors plus a zero-value sink.

    query: [batch, sequence, heads, width]; kv: [batch, cache, width]; indices:
    [batch, sequence, selected]; sink: FP32 [heads]. Negative indices are padding;
    nonnegative IDs must index cache entries. Callers own selection/causality and
    finite input validation. Repeated IDs intentionally count repeatedly.

    Temporary tensors scale with batch, query_chunk and key_chunk, not the full
    sequence/selected lengths. No whole-cache FP32 cast or host index extraction.
    """
    if (not isinstance(query, torch.Tensor) or query.ndim != 4
            or query.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('queries must be FP16/BF16/FP32 [batch, sequence, heads, width]')
    batch, sequence, heads, width = query.shape
    if (heads < 1 or width < 1 or not isinstance(kv, torch.Tensor) or kv.ndim != 3
            or kv.shape[0] != batch or kv.shape[2] != width or kv.shape[1] < 1
            or kv.dtype != query.dtype or kv.device != query.device):
        raise ValueError('shared KV must match query batch, width, dtype and device with nonempty cache')
    if (not isinstance(indices, torch.Tensor) or indices.ndim != 3 or indices.shape[:2] != (batch, sequence)
            or indices.dtype not in (torch.int32, torch.int64) or indices.device != query.device):
        raise ValueError('selected indices must be same-device integer [batch, sequence, selected]')
    if (not isinstance(sink, torch.Tensor) or sink.shape != (heads,)
            or sink.dtype != torch.float32 or sink.device != query.device):
        raise ValueError('attention sink must be same-device FP32 [heads]')
    if (type(query_chunk) is not int or not 1 <= query_chunk <= 128
            or type(key_chunk) is not int or not 1 <= key_chunk <= 256):
        raise ValueError('query/key chunks must be integers in 1..128/1..256')
    scale = width**-.5 if scale is None else scale
    if type(scale) not in (int, float) or not 0 < scale <= torch.finfo(torch.float32).max:
        raise ValueError('attention scale must be finite and positive')
    output = torch.empty_like(query)
    batch_ids = torch.arange(batch, device=query.device)[:, None, None]
    for start in range(0, sequence, query_chunk):
        stop = min(start+query_chunk, sequence)
        q = query[:, start:stop].float()
        maximum = sink.reshape(1, 1, heads).expand(batch, stop-start, heads)
        denominator = torch.ones_like(maximum)
        accumulator = torch.zeros_like(q)
        for key_start in range(0, indices.shape[-1], key_chunk):
            selected = indices[:, start:stop, key_start:key_start+key_chunk].long()
            valid = selected >= 0
            vectors = kv[batch_ids, selected.clamp_min(0)].float()
            # Padding must not read poisoned/uninitialized slot zero into a matmul.
            vectors = torch.where(valid.unsqueeze(-1), vectors, 0.)
            scores = torch.matmul(q, vectors.transpose(-1, -2))*scale
            scores = scores.masked_fill(~valid.unsqueeze(-2), -float('inf'))
            new_maximum = torch.maximum(maximum, scores.amax(-1))
            old_factor = (maximum-new_maximum).exp()
            probabilities = (scores-new_maximum.unsqueeze(-1)).exp()
            accumulator = accumulator*old_factor.unsqueeze(-1) + torch.matmul(probabilities, vectors)
            denominator = denominator*old_factor + probabilities.sum(-1)
            maximum = new_maximum
        output[:, start:stop] = (accumulator/denominator.unsqueeze(-1)).to(query.dtype)
    return output
