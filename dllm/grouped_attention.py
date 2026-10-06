"""Bounded FP32 causal prefill that reuses each KV matrix across query heads.

Flatten grouped query heads into GEMM rows instead of replicating long KV
matrices. Partition query rows to bound score/probability scratch; each query
still sees its complete causal prefix. This does not change KV/state precision.
"""
import torch


def use_grouped_prefill(accelerator, precision, prefill, query_tokens, kv_tokens):
    return (accelerator == 'rocm' and precision == 'compensated' and prefill
            and 1 < query_tokens <= kv_tokens and kv_tokens >= 1024)


def attention(q, k, v, *, scratch_bytes=256*1024**2):
    if (q.ndim != 4 or k.ndim != 4 or v.shape != k.shape
            or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
            or not q.shape[0] or not k.shape[1] or not q.shape[1]
            or q.shape[1] % k.shape[1] or not q.shape[-1]
            or not 0 < q.shape[2] <= k.shape[2]
            or any(x.dtype != torch.float32 or x.device != q.device for x in (q, k, v))):
        raise ValueError('requires nonempty matching FP32 causal GQA tensors')
    if type(scratch_bytes) is not int or scratch_bytes <= 0:
        raise ValueError('attention scratch budget must be a positive byte count')
    batch, heads, length, dim = q.shape
    repeats = heads // k.shape[1]
    context = k.shape[2]
    # Two FP32 score/probability buffers and one shared boolean mask per row.
    rows = scratch_bytes // (context*(2*batch*repeats*4+1))
    if rows < 1:
        raise ValueError('attention scratch budget cannot fit one query row')
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    keys = torch.arange(context, device=q.device)[None, :]
    prefix = context-length
    for start in range(0, length, rows):
        stop = min(start+rows, length)
        mask = keys > (prefix+torch.arange(start, stop, device=q.device)[:, None])
        for head in range(k.shape[1]):
            first, last = head*repeats, (head+1)*repeats
            query = q[:, first:last, start:stop].reshape(batch, repeats*(stop-start), dim)
            scores = query @ k[:, head].transpose(-1, -2)
            scores.mul_(dim**-.5)
            scores = scores.reshape(batch, repeats, stop-start, context)
            scores.masked_fill_(mask, -float('inf'))
            probabilities = scores.softmax(-1).reshape(batch, repeats*(stop-start), context)
            output[:, first:last, start:stop] = (probabilities @ v[:, head]).reshape(batch, repeats, stop-start, dim)
            del query, scores, probabilities
        del mask
    return output
