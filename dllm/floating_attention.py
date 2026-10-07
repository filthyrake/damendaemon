"""Complete floating DeepSeek attention prefill; not a serving admission path.

Prepared floating projection weights replace packed GEMMs for component checks.
Reference accelerator rounding, packed arithmetic and model/GPU qualification
remain separate. Parameter names are relative to a checkpoint attention module.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .activation_simulation import simulate_fp8
from .cache_indices import prefill_indices
from .compressed_pool import PoolState
from .deepseek_position import rotate_pairs
from .floating_compressor import compress_hidden
from .floating_indexer import index_hidden
from .selected_attention import selected_attention


@dataclass(frozen=True)
class AttentionPrefillState:
    position: int
    window: torch.Tensor
    compressed: torch.Tensor
    pool: PoolState | None
    index_keys: torch.Tensor | None
    index_pool: PoolState | None


def _norm(value, weight, epsilon):
    fp32 = value.float()
    return (weight*(fp32*(fp32.square().mean(-1, keepdim=True)+epsilon).rsqrt())).to(value.dtype)


def _check(hidden, weights, freq, groups, window, ratio, top_k, chunk, epsilon):
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or min(hidden.shape) < 1
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not isinstance(weights, dict)):
        raise ValueError('floating attention requires nonempty floating hidden state and a weight mapping')
    if (type(groups) is not int or groups < 1 or type(window) is not int or not 1 <= window < 2**31
            or type(ratio) is not int or ratio not in (0, 4, 128)
            or type(top_k) is not int or not 1 <= top_k <= 4096
            or type(chunk) is not int or not 1 <= chunk <= 128
            or type(epsilon) not in (int, float) or not 0 < epsilon <= torch.finfo(torch.float32).max):
        raise ValueError('invalid floating attention geometry/options')
    for name in ('q_norm.weight', 'kv_norm.weight', 'attn_sink'):
        value = weights.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < 1:
            raise ValueError('missing or invalid attention normalization/sink geometry')
    heads, width, qrank = (weights[n].numel() for n in ('attn_sink', 'kv_norm.weight', 'q_norm.weight'))
    output = weights.get('wo_a.weight')
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] < 1 or output.shape[0] % groups or heads % groups:
        raise ValueError('invalid grouped output projection geometry')
    orank = output.shape[0]//groups
    if (not isinstance(freq, torch.Tensor) or freq.ndim != 1 or freq.dtype != torch.float32
            or freq.device != hidden.device or not 1 <= freq.numel() <= width//2
            or (width-2*freq.numel()) % 64):
        raise ValueError('rotary suffix must leave complete 64-value KV simulation blocks')
    dim = hidden.shape[-1]
    expected = {'q_norm.weight':((qrank,), torch.float32), 'kv_norm.weight':((width,), torch.float32),
                'attn_sink':((heads,), torch.float32), 'wq_a.weight':((qrank, dim), hidden.dtype),
                'wq_b.weight':((heads*width, qrank), hidden.dtype), 'wkv.weight':((width, dim), hidden.dtype),
                'wo_a.weight':((groups*orank, heads*width//groups), hidden.dtype),
                'wo_b.weight':((dim, groups*orank), hidden.dtype)}
    def compressor(prefix, output_width):
        channels = output_width*(2 if ratio == 4 else 1)
        for suffix, shape in (('wkv.weight',(channels,dim)), ('wgate.weight',(channels,dim)),
                              ('ape',(ratio,channels)), ('norm.weight',(output_width,))):
            expected[prefix+'.'+suffix] = (shape, torch.float32)
    index_width = None
    if ratio:
        compressor('compressor', width)
    if ratio == 4:
        norm, head_weight = weights.get('indexer.compressor.norm.weight'), weights.get('indexer.weights_proj.weight')
        if not isinstance(norm, torch.Tensor) or norm.ndim != 1 or not isinstance(head_weight, torch.Tensor) or head_weight.ndim != 2:
            raise ValueError('missing indexer geometry')
        index_width, index_heads = norm.numel(), head_weight.shape[0]
        if not 32 <= index_width <= 4096 or index_width & (index_width-1) or index_width < 2*freq.numel() or index_heads < 1:
            raise ValueError('invalid indexer width/head geometry')
        expected['indexer.wq_b.weight'] = ((index_heads*index_width,qrank),hidden.dtype)
        expected['indexer.weights_proj.weight'] = ((index_heads,dim),hidden.dtype)
        compressor('indexer.compressor', index_width)
    if set(weights) != set(expected):
        raise ValueError('floating attention parameter names do not match the declared variant')
    for name, (shape, dtype) in expected.items():
        value = weights[name]
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != hidden.device:
            raise ValueError(f'floating attention parameter shape/dtype/device mismatch: {name}')
    return heads, width, orank, index_width


@torch.no_grad()
def attention_prefill(hidden, weights, inverse_frequencies, *, groups, window, ratio=0,
                      index_top_k=512, query_chunk=16, epsilon=1e-6):
    """Return projected output and compact prepared caches for a fresh sequence.

    Two bounded passes avoid retaining full-sequence multi-head queries. Persistent
    results own the ring allocation and compressed vectors; they do not retain a
    view of the entire raw prefill KV. This API does not yet execute cached decode.
    Caller supplies the appropriate plain/compressed rotary frequency schedule.
    """
    heads, width, orank, index_width = _check(hidden, weights, inverse_frequencies, groups, window, ratio,
                                           index_top_k, query_chunk, epsilon)
    batch, length = hidden.shape[:2]
    rd = inverse_frequencies.numel()*2
    kv = hidden.new_empty(batch, length, width)
    compressed = hidden.new_empty(batch, length//ratio if ratio else 0, width)
    index_keys = hidden.new_empty(batch, length//ratio, index_width) if ratio == 4 else None
    pool = index_pool = None
    positions = torch.arange(length, device=hidden.device, dtype=torch.int64)
    def compress(part, prefix, state, mode):
        return compress_hidden(part, weights[prefix+'.wkv.weight'], weights[prefix+'.wgate.weight'],
                               weights[prefix+'.ape'], weights[prefix+'.norm.weight'], inverse_frequencies,
                               state=state, epsilon=epsilon, cache_transform=mode)
    for start in range(0, length, query_chunk):
        stop = min(length, start+query_chunk)
        part = hidden[:, start:stop]
        vectors = _norm(F.linear(part, weights['wkv.weight']), weights['kv_norm.weight'], epsilon)
        vectors[..., -rd:] = rotate_pairs(vectors[..., -rd:], positions[start:stop], inverse_frequencies)
        if width > rd:
            vectors[..., :-rd] = simulate_fp8(vectors[..., :-rd])
        kv[:, start:stop] = vectors
        if ratio:
            ready, pool, _ = compress(part, 'compressor', pool, 'kv_fp8')
            compressed[:, start//ratio:stop//ratio] = ready
        if ratio == 4:
            ready, index_pool, _ = compress(part, 'indexer.compressor', index_pool, 'index_fp4')
            index_keys[:, start//ratio:stop//ratio] = ready
    combined = torch.cat((kv, compressed), 1) if ratio else kv
    output = torch.empty_like(hidden)
    for start in range(0, length, query_chunk):
        stop = min(length, start+query_chunk)
        part, pos = hidden[:, start:stop], positions[start:stop]
        low_rank = _norm(F.linear(part, weights['wq_a.weight']), weights['q_norm.weight'], epsilon)
        query = F.linear(low_rank, weights['wq_b.weight']).unflatten(-1, (heads, width))
        query = query*(query.square().mean(-1, keepdim=True)+epsilon).rsqrt()
        query[..., -rd:] = rotate_pairs(query[..., -rd:], pos, inverse_frequencies)
        indices = prefill_indices(length, window=window, ratio=ratio if ratio != 4 else 0,
                                  batch=batch, query_start=start, query_stop=stop, device=hidden.device)
        if ratio == 4:
            selected, _ = index_hidden(part, low_rank, weights['indexer.wq_b.weight'], weights['indexer.weights_proj.weight'],
                                       index_keys, pos.unsqueeze(0).expand(batch, -1), inverse_frequencies,
                                       top_k=index_top_k, offset=length, query_chunk=query_chunk)
            indices = torch.cat((indices, selected), -1)
        attended = selected_attention(query, combined, indices, weights['attn_sink'], query_chunk=query_chunk)
        attended[..., -rd:] = rotate_pairs(attended[..., -rd:], pos, inverse_frequencies, inverse=True)
        grouped = attended.reshape(batch, stop-start, groups, heads*width//groups)
        projected = torch.einsum('bsgd,grd->bsgr', grouped, weights['wo_a.weight'].reshape(groups, orank, -1))
        output[:, start:stop] = F.linear(projected.flatten(2), weights['wo_b.weight'])
    ring = hidden.new_zeros(batch, window, width)
    first = max(0, length-window)
    ring.index_copy_(1, positions[first:] % window, kv[:, first:])
    return output, AttentionPrefillState(length, ring, compressed, pool, index_keys, index_pool)
