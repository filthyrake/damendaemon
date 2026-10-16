"""Floating DeepSeek attention prefill and cached decode; not a serving admission path.

Projection weights may be floating tensors or explicit CPU packed adapters.
Reference accelerator rounding, packed arithmetic and model/GPU qualification
remain separate. Parameter names are relative to a checkpoint attention module.
"""
from dataclasses import dataclass, field
import threading

import torch

from .activation_simulation import simulate_fp8
from .cache_indices import prefill_indices, ring_decode_indices
from .compressed_pool import PoolState
from .deepseek_position import rotate_pairs
from .floating_compressor import checked_compress_hidden as compress_hidden
from .floating_indexer import checked_index_hidden as index_hidden
from .prepared_projection import PackedProjection, is_projection, linear
from .selected_attention import selected_attention
from .reference_numerics import finite


@dataclass
class AttentionState:
    position: int
    cache: torch.Tensor
    window_size: int
    pool: PoolState | None
    index_keys: torch.Tensor | None
    index_pool: PoolState | None
    capacity: int
    ratio: int
    groups: int
    index_top_k: int
    epsilon: float
    weights: dict = field(repr=False)
    frequencies: torch.Tensor = field(repr=False)
    max_cache_bytes: int
    bytes_per_slot: int
    poisoned: bool = False
    closed: bool = False
    lock: object = field(default_factory=threading.RLock, repr=False)
    backend: object = field(default=None,repr=False)

    @property
    def window(self):
        return None if self.cache is None else self.cache[:, :self.window_size]

    @property
    def compressed(self):
        return None if self.cache is None else self.cache[:, self.window_size:]

    def _usable(self):
        if self.closed or self.poisoned:
            raise RuntimeError('attention cache is closed or poisoned')

    def reorder(self, indices):
        """Copy/reorder slots atomically; all slots share the logical position."""
        with self.lock:
            self._usable()
            if (not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.numel() < 1
                    or indices.dtype != torch.int64 or indices.device != self.cache.device
                    or indices.numel()*self.bytes_per_slot > self.max_cache_bytes):
                raise ValueError('invalid slot order or cache budget exceeded')
            if self.backend is not None:self.backend.admit_reorder(self,indices)
            cache = self.cache.index_select(0, indices)
            index_keys = self.index_keys.index_select(0, indices) if self.index_keys is not None else None
            pool = self.pool.reorder(indices) if self.pool is not None else None
            index_pool = self.index_pool.reorder(indices) if self.index_pool is not None else None
            self.cache, self.index_keys, self.pool, self.index_pool = cache, index_keys, pool, index_pool

    def close(self):
        """Release this component's references; caller-owned weights remain owned by caller."""
        with self.lock:
            self.cache = self.index_keys = self.pool = self.index_pool = None
            self.weights = {}
            self.frequencies = None
            self.closed = True
            self.backend = None


def _norm(value, weight, epsilon):
    fp32 = finite(value.float(), 'normalization input')
    finite(weight, 'normalization weight')
    variance = finite(fp32.square().mean(-1, keepdim=True)+epsilon, 'normalization variance')
    return finite((weight*(fp32*variance.rsqrt())).to(value.dtype), 'normalization output')


def _check(hidden, weights, freq, groups, window, ratio, top_k, chunk, epsilon,backend=None):
    projection_check=is_projection if backend is None else backend.projection
    packed_types=PackedProjection if backend is None else (PackedProjection,backend.packed_type)
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
    if not projection_check(output) or output.shape[0] < 1 or output.shape[0] % groups or heads % groups:
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
        if not isinstance(norm, torch.Tensor) or norm.ndim != 1 or not projection_check(head_weight):
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
        packed_allowed = name in ('wq_a.weight','wq_b.weight','wkv.weight','wo_a.weight','wo_b.weight',
                                   'indexer.wq_b.weight','indexer.weights_proj.weight')
        valid_type = isinstance(value,torch.Tensor) or (packed_allowed and isinstance(value,packed_types))
        if not valid_type or value.shape != shape or value.dtype != dtype or value.device != hidden.device:
            raise ValueError(f'floating attention parameter shape/dtype/device mismatch: {name}')
    return heads, width, orank, index_width


def _grouped_project(grouped, weight, groups, orank,backend=None):
    packed_types=PackedProjection if backend is None else (PackedProjection,backend.packed_type)
    project=linear if backend is None else backend.linear
    if isinstance(weight,packed_types):
        return torch.stack([project(grouped[:,:,group],weight.rows(group*orank,(group+1)*orank))
                            for group in range(groups)],dim=2)
    return torch.einsum('bsgd,grd->bsgr',grouped,weight.reshape(groups,orank,-1))


def _query_norm(query,epsilon,backend):
    variance=query.square().mean(-1,keepdim=True)+epsilon
    if backend is not None:finite(query,'CUDA attention query');finite(variance,'CUDA query variance')
    result=query*variance.rsqrt()
    return finite(result,'CUDA normalized query') if backend is not None else result


def cache_bytes_per_slot(window,ratio,capacity,width,index_width,item_bytes):
    """Storage reservation for an already validated attention geometry."""
    if type(capacity) is not int or not 1<=capacity<2**31:
        raise ValueError('invalid attention cache capacity')
    blocks = capacity//ratio if ratio else 0
    bytes_per_slot = (window+blocks)*width*item_bytes
    if ratio:
        factor = 2 if ratio == 4 else 1
        bytes_per_slot += 2*factor*ratio*factor*width*4
    if ratio == 4:
        bytes_per_slot += blocks*index_width*item_bytes+2*2*ratio*2*index_width*4
    return bytes_per_slot


@torch.no_grad()
def _attention_prefill(hidden, weights, inverse_frequencies, *, groups, window, ratio=0,
                      index_top_k=512, query_chunk=16, epsilon=1e-6, max_tokens=None,
                      max_cache_bytes=256*1024*1024,backend=None):
    """Return projected output and compact prepared caches for a fresh sequence.

    Two bounded passes avoid retaining full-sequence multi-head queries. Persistent
    results own the ring allocation and compressed vectors; they do not retain a
    view of the entire raw prefill KV. max_tokens reserves decode capacity (default:
    prefill length). Cache budget includes pool buffers, excluding weights, output
    and transient prefill/workspace tensors. Caller supplies the correct frequency
    schedule and must keep prepared weights/frequencies immutable while state lives.
    Device synchronization/lifetime qualification remains separate from CPU proof.
    """
    heads, width, orank, index_width = _check(hidden, weights, inverse_frequencies, groups, window, ratio,
                                           index_top_k, query_chunk, epsilon,backend)
    linear_fn=linear if backend is None else backend.linear
    compress_fn=compress_hidden if backend is None else backend.compress
    index_fn=index_hidden if backend is None else backend.index
    selected_fn=selected_attention if backend is None else backend.selected
    batch, length = hidden.shape[:2]
    capacity = length if max_tokens is None else max_tokens
    if (type(capacity) is not int or not length <= capacity < 2**31
            or type(max_cache_bytes) is not int or max_cache_bytes < 1):
        raise ValueError('invalid attention cache capacity/budget')
    blocks = capacity//ratio if ratio else 0
    bytes_per_slot = cache_bytes_per_slot(window,ratio,capacity,width,index_width,hidden.element_size())
    if batch*bytes_per_slot > max_cache_bytes:
        raise ValueError('attention cache reservation exceeds budget')
    rd = inverse_frequencies.numel()*2
    cache = hidden.new_zeros(batch, window+blocks, width)
    kv = hidden.new_empty(batch, length, width)
    compressed = cache[:, window:]
    index_keys = hidden.new_zeros(batch, blocks, index_width) if ratio == 4 else None
    pool = index_pool = None
    positions = torch.arange(length, device=hidden.device, dtype=torch.int64)
    def compress(part, prefix, state, mode):
        return compress_fn(part, weights[prefix+'.wkv.weight'], weights[prefix+'.wgate.weight'],
                               weights[prefix+'.ape'], weights[prefix+'.norm.weight'], inverse_frequencies,
                               state=state, epsilon=epsilon, cache_transform=mode)
    for start in range(0, length, query_chunk):
        stop = min(length, start+query_chunk)
        part = hidden[:, start:stop]
        vectors = _norm(linear_fn(part, weights['wkv.weight']), weights['kv_norm.weight'], epsilon)
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
    combined = torch.cat((kv, compressed[:, :length//ratio]), 1) if ratio else kv
    output = torch.empty_like(hidden)
    for start in range(0, length, query_chunk):
        stop = min(length, start+query_chunk)
        part, pos = hidden[:, start:stop], positions[start:stop]
        low_rank = _norm(linear_fn(part, weights['wq_a.weight']), weights['q_norm.weight'], epsilon)
        query = linear_fn(low_rank, weights['wq_b.weight']).unflatten(-1, (heads, width))
        query = _query_norm(query,epsilon,backend)
        query[..., -rd:] = rotate_pairs(query[..., -rd:], pos, inverse_frequencies)
        indices = prefill_indices(length, window=window, ratio=ratio if ratio != 4 else 0,
                                  batch=batch, query_start=start, query_stop=stop, device=hidden.device)
        if ratio == 4:
            selected, _ = index_fn(part, low_rank, weights['indexer.wq_b.weight'], weights['indexer.weights_proj.weight'],
                                       index_keys[:, :length//ratio], pos.unsqueeze(0).expand(batch, -1), inverse_frequencies,
                                       top_k=index_top_k, offset=length, query_chunk=query_chunk)
            indices = torch.cat((indices, selected), -1)
        attended = selected_fn(query, combined, indices, weights['attn_sink'], query_chunk=query_chunk)
        attended[..., -rd:] = rotate_pairs(attended[..., -rd:], pos, inverse_frequencies, inverse=True)
        grouped = attended.reshape(batch, stop-start, groups, heads*width//groups)
        projected = _grouped_project(grouped,weights['wo_a.weight'],groups,orank,backend)
        output[:, start:stop] = linear_fn(projected.flatten(2), weights['wo_b.weight'])
    ring = cache[:, :window]
    first = max(0, length-window)
    ring.index_copy_(1, positions[first:] % window, kv[:, first:])
    return output, AttentionState(length, cache, window, pool, index_keys, index_pool, capacity,
                                        ratio, groups, index_top_k, epsilon, dict(weights), inverse_frequencies,
                                        max_cache_bytes, bytes_per_slot,backend=backend)


@torch.no_grad()
def _attention_decode(hidden, state, *, position):
    """Advance one token per slot; a failed mutated state cannot be reused.

    Positions are explicit to reject duplicate/stale updates. Shared state access
    is serialized. Prepared weights/frequencies are retained by the state; no
    alternate model parameters may be supplied to this decode operation.
    """
    if not isinstance(state, AttentionState):
        raise ValueError('attention decode requires prepared cache state')
    with state.lock:
        state._usable()
        if type(position) is not int or position != state.position or position >= state.capacity:
            raise ValueError('stale position or attention cache capacity exhausted')
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[1] != 1
                or hidden.shape[0] != state.cache.shape[0] or hidden.device != state.cache.device
                or hidden.dtype != state.cache.dtype):
            raise ValueError('decode must provide one matching token per prepared slot')
        backend=state.backend
        linear_fn=linear if backend is None else backend.linear
        compress_fn=compress_hidden if backend is None else backend.compress
        index_fn=index_hidden if backend is None else backend.index
        selected_fn=selected_attention if backend is None else backend.selected
        weights, freq, epsilon = state.weights, state.frequencies, state.epsilon
        heads, width, orank, _ = _check(hidden, weights, freq, state.groups, state.window_size,
                                      state.ratio, state.index_top_k, 1, epsilon,backend)
        batch, rd = hidden.shape[0], freq.numel()*2
        pos = torch.tensor([position], dtype=torch.int64, device=hidden.device)
        low_rank = _norm(linear_fn(hidden, weights['wq_a.weight']), weights['q_norm.weight'], epsilon)
        query = linear_fn(low_rank, weights['wq_b.weight']).unflatten(-1, (heads,width))
        query = _query_norm(query,epsilon,backend)
        query[..., -rd:] = rotate_pairs(query[..., -rd:], pos, freq)
        vectors = _norm(linear_fn(hidden, weights['wkv.weight']), weights['kv_norm.weight'], epsilon)
        vectors[..., -rd:] = rotate_pairs(vectors[..., -rd:], pos, freq)
        if width > rd:
            vectors[..., :-rd] = simulate_fp8(vectors[..., :-rd])
        def compress(prefix, pool, mode):
            return compress_fn(hidden, weights[prefix+'.wkv.weight'], weights[prefix+'.wgate.weight'],
                                   weights[prefix+'.ape'], weights[prefix+'.norm.weight'], freq,
                                   state=pool, epsilon=epsilon, cache_transform=mode)
        pool, index_pool = state.pool, state.index_pool
        if state.ratio:
            ready, pool, _ = compress('compressor', pool, 'kv_fp8')
        if state.ratio == 4:
            index_ready, index_pool, _ = compress('indexer.compressor', index_pool, 'index_fp4')
        # Everything above is pure preparation. Mutations below poison on failure.
        try:
            state.window[:, position % state.window_size] = vectors[:, 0]
            visible = (position+1)//state.ratio if state.ratio else 0
            if state.ratio and ready.shape[1]:
                state.compressed[:, visible-1:visible] = ready
            if state.ratio == 4 and index_ready.shape[1]:
                state.index_keys[:, visible-1:visible] = index_ready
            indices = ring_decode_indices([position]*batch, window=state.window_size,
                                          ratio=state.ratio if state.ratio != 4 else 0, device=hidden.device)
            if state.ratio == 4:
                selected, _ = index_fn(hidden, low_rank, weights['indexer.wq_b.weight'], weights['indexer.weights_proj.weight'],
                                           state.index_keys[:, :visible], pos.unsqueeze(0).expand(batch,-1), freq,
                                           top_k=state.index_top_k, offset=state.window_size, query_chunk=1)
                indices = torch.cat((indices, selected), -1)
            attended = selected_fn(query, state.cache, indices, weights['attn_sink'], query_chunk=1)
            attended[..., -rd:] = rotate_pairs(attended[..., -rd:], pos, freq, inverse=True)
            grouped = attended.reshape(batch, 1, state.groups, heads*width//state.groups)
            projected = _grouped_project(grouped,weights['wo_a.weight'],state.groups,orank,backend)
            output = linear_fn(projected.flatten(2), weights['wo_b.weight'])
        except BaseException:
            state.poisoned = True
            raise
        state.pool, state.index_pool, state.position = pool, index_pool, position+1
        return output


def attention_prefill(hidden, weights, inverse_frequencies, *, groups, window, ratio=0,
                      index_top_k=512, query_chunk=16, epsilon=1e-6, max_tokens=None,
                      max_cache_bytes=256*1024*1024):
    return _attention_prefill(hidden,weights,inverse_frequencies,groups=groups,window=window,ratio=ratio,
                              index_top_k=index_top_k,query_chunk=query_chunk,epsilon=epsilon,
                              max_tokens=max_tokens,max_cache_bytes=max_cache_bytes)


def attention_decode(hidden,state,*,position):
    if isinstance(state,AttentionState) and state.backend is not None:
        from .cuda_floating_attention import attention_decode as cuda_decode
        return cuda_decode(hidden,state,position=position)
    return _attention_decode(hidden,state,position=position)
