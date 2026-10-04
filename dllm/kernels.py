"""Inference kernels with FP32 reductions and optional small-batch GEMV.

Dense FP16 matrix operations use cuBLAS. Gemma can retain FP32 arithmetic
while reading FP16 weights directly in its decode matrix-vector kernel.

Qwen2/Llama preserve FP16 rounding points. Qwen3 retains FP32 elementwise
intermediates before storing FP16 inputs for the next matrix operation.
Decode attention uses a live-length mask and splits long contexts into blocks.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr, PRECISE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + cols, cols < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    norm = x * inv
    if not PRECISE:
        norm = norm.to(tl.float16).to(tl.float32)
    weight = tl.load(W + cols, cols < D, other=0).to(tl.float32)
    tl.store(Y + row * D + cols, norm * weight, cols < D)


def rms(x, weight, eps, precise=False):
    out = torch.empty(x.shape, dtype=weight.dtype, device=x.device)
    _rms[(x.numel() // x.shape[-1],)](x, weight, out, x.shape[-1], eps,
                                   tr.next_power_of_2(x.shape[-1]), precise)
    return out


@tr.jit
def _swiglu(X, Y, D: tl.constexpr, BLOCK: tl.constexpr, PRECISE: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * 2 * D + cols, cols < D, other=0).to(tl.float32)
    up = tl.load(X + row * 2 * D + D + cols, cols < D, other=0).to(tl.float32)
    activated = gate / (1 + tl.exp(-gate))
    if not PRECISE:
        activated = activated.to(tl.float16).to(tl.float32)
    tl.store(Y + row * D + cols, activated * up, cols < D)


def swiglu(gate_up, precise=False):
    rows, width = gate_up.shape
    out = torch.empty((rows, width // 2), dtype=gate_up.dtype, device=gate_up.device)
    _swiglu[(rows, tr.cdiv(width // 2, 256))](gate_up, out, width // 2, 256, precise)
    return out


@tr.jit
def _rope_store(X, COS, SIN, POS, Q, K, V, QW, KW,
                NORM: tl.constexpr, EPS: tl.constexpr, PRECISE: tl.constexpr,
                T: tl.constexpr, H: tl.constexpr, KH: tl.constexpr,
                D: tl.constexpr, C: tl.constexpr, BD: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BD)
    position = tl.load(POS + token)
    base = token * (H + 2 * KH) * D + head * D
    x = tl.load(X + base + cols, cols < D, other=0).to(tl.float32)
    if head < H + KH:
        if NORM:
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            weight = tl.load((QW if head < H else KW) + cols, cols < D, other=0).to(tl.float32)
            x = x * inv * weight
            other_cols = (cols + D // 2) % D
            other = tl.load(X + base + other_cols, cols < D, other=0).to(tl.float32)
            other_weight = tl.load((QW if head < H else KW) + other_cols, cols < D, other=0).to(tl.float32)
            other = other * inv * other_weight
        else:
            other = tl.load(X + base + (cols + D // 2) % D, cols < D, other=0).to(tl.float32)
        other = tl.where(cols < D // 2, -other, other)
        cos = tl.load(COS + position * D + cols, cols < D, other=0).to(tl.float32)
        sin = tl.load(SIN + position * D + cols, cols < D, other=0).to(tl.float32)
        a, b = x * cos, other * sin
        if not NORM and not PRECISE:
            a = a.to(tl.float16).to(tl.float32)
            b = b.to(tl.float16).to(tl.float32)
        rotated = a + b
        if head < H:
            tl.store(Q + (head * T + token) * D + cols, rotated, cols < D)
        else:
            tl.store(K + ((head - H) * C + position) * D + cols, rotated, cols < D)
    else:
        tl.store(V + ((head - H - KH) * C + position) * D + cols, x, cols < D)



def rope_store(qkv, cos, sin, positions, cache_k, cache_v, heads, q_norm=None, k_norm=None, eps=1e-6, precise=False):
    tokens = qkv.shape[0]
    _, kv_heads, context, dim = cache_k.shape
    query = torch.empty((1, heads, tokens, dim), dtype=qkv.dtype, device=qkv.device)
    _rope_store[(tokens, heads + 2 * kv_heads)](
        qkv, cos, sin, positions, query, cache_k, cache_v,
        qkv if q_norm is None else q_norm, qkv if k_norm is None else k_norm,
        q_norm is not None, eps, precise, tokens, heads, kv_heads, dim, context, tr.next_power_of_2(dim), enable_fp_fusion=False)
    return query


@tr.jit
def _attention(Q, K, V, POS, SLOTS, OUT, PART, LSE, ALIBI, HAS_ALIBI: tl.constexpr,
               PAGES, PAGED: tl.constexpr, PAGE_SIZE: tl.constexpr, MAX_PAGES: tl.constexpr,
               H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
               C: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr, SLOTTED: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr, BD: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    position = tl.load(POS + batch)
    slot = tl.load(SLOTS + batch) if SLOTTED else 0
    cols = tl.arange(0, BD)
    if split * BLOCK <= position and (WINDOW == 0 or (split + 1) * BLOCK > position - WINDOW + 1):
        times = split * BLOCK + tl.arange(0, BLOCK)
        valid = (times <= position) & (times < C)
        if WINDOW:
            valid = valid & (times > position - WINDOW)
        q = tl.load(Q + bh * D + cols, cols < D, other=0).to(tl.float32)
        if PAGED:
            physical = tl.load(PAGES + slot * MAX_PAGES + times // PAGE_SIZE, valid, other=0)
            kh = physical * KH + head // (H // KH)
            offsets = (kh[:, None] * PAGE_SIZE + (times % PAGE_SIZE)[:, None]) * D + cols[None, :]
        else:
            kh = slot * KH + head // (H // KH)
            offsets = (kh * C + times[:, None]) * D + cols[None, :]
        k = tl.load(K + offsets, valid[:, None] & (cols[None, :] < D), other=0).to(tl.float32)
        logits = tl.sum(k * q[None, :], 1) * SCALE
        if HAS_ALIBI:
            logits += tl.load(ALIBI + head) * times
        logits = tl.where(valid, logits, -float("inf"))
        maximum = tl.max(logits, 0)
        weights = tl.exp(logits - maximum)
        denominator = tl.sum(weights, 0)
        v = tl.load(V + offsets, valid[:, None] & (cols[None, :] < D), other=0).to(tl.float32)
        result = tl.sum(v * weights[:, None], 0) / denominator
        if SPLITS == 1:
            tl.store(OUT + bh * D + cols, result, cols < D)
        else:
            tl.store(PART + (bh * SPLITS + split) * D + cols, result, cols < D)
            tl.store(LSE + bh * SPLITS + split, maximum + tl.log(denominator))
    else:
        if SPLITS > 1:
            tl.store(PART + (bh * SPLITS + split) * D + cols, 0., cols < D)
            tl.store(LSE + bh * SPLITS + split, -float("inf"))


@tr.jit
def _merge(PART, LSE, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr, BD: tl.constexpr):
    head = tl.program_id(0)
    splits = tl.arange(0, BLOCK)
    cols = tl.arange(0, BD)
    lse = tl.load(LSE + head * S + splits, splits < S, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    weights = weights / tl.sum(weights, 0)
    partial = tl.load(PART + (head * S + splits[:, None]) * D + cols[None, :],
                      (splits[:, None] < S) & (cols[None, :] < D), other=0)
    tl.store(OUT + head * D + cols, tl.sum(partial * weights[:, None], 0), cols < D)


def attention(query, cache_k, cache_v, positions, slots=None, window=0, scale=None, alibi=None, page_table=None):
    batch, heads, _, dim = query.shape
    if alibi is not None and (alibi.shape != (heads,) or alibi.dtype != torch.float32
                              or alibi.device != query.device or not alibi.is_contiguous()):
        raise ValueError('ALiBi needs one contiguous FP32 slope per query head on the same device')
    _, kv_heads, context, _ = cache_k.shape
    page_size = context
    max_pages = 0 if page_table is None else page_table.shape[1]
    if page_table is not None:
        context = max_pages * page_size
    block = min(256, tr.next_power_of_2(context))
    splits = tr.cdiv(context, block)
    out = torch.empty_like(query)
    partial = torch.empty((batch * heads, splits, dim), device=query.device, dtype=torch.float32)
    lse = torch.empty((batch * heads, splits), device=query.device, dtype=torch.float32)
    _attention[(batch * heads, splits)](query, cache_k, cache_v, positions,
                               positions if slots is None else slots, out, partial, lse,
                               positions if alibi is None else alibi, alibi is not None,
                               positions if page_table is None else page_table, page_table is not None, page_size, max_pages,
                               heads, kv_heads, dim, context, splits, block, slots is not None, window, dim ** -.5 if scale is None else scale, tr.next_power_of_2(dim), num_warps=4)
    if splits > 1:
        _merge[(batch * heads,)](partial, lse, out, dim, splits, tr.next_power_of_2(splits), tr.next_power_of_2(dim))
    return out


@tr.jit
def _rope_batch(X, COS, SIN, POS, SLOTS, Q, K, V, QW, KW, TMPK, TMPV, MATERIALIZE: tl.constexpr,
                PAGES, PAGED: tl.constexpr, MAX_PAGES: tl.constexpr,
                NORM: tl.constexpr, EPS: tl.constexpr, PRECISE: tl.constexpr,
                H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, RD: tl.constexpr, C: tl.constexpr, BD: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BD)
    position, slot = tl.load(POS + batch), tl.load(SLOTS + batch)
    cache_slot, cache_position = slot, position
    if PAGED:
        cache_slot = tl.load(PAGES + slot * MAX_PAGES + position // C)
        cache_position = position % C
    base = batch * (H + 2 * KH) * D + head * D
    x = tl.load(X + base + cols, cols < D, other=0).to(tl.float32)
    if head < H + KH:
        if NORM:
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            weight = tl.load((QW if head < H else KW) + cols, cols < D, other=0).to(tl.float32)
            x = x * inv * weight
            other_cols = (cols + RD // 2) % RD
            other = tl.load(X + base + other_cols, cols < D, other=0).to(tl.float32)
            other_weight = tl.load((QW if head < H else KW) + other_cols, cols < D, other=0).to(tl.float32)
            other = other * inv * other_weight
        else:
            other = tl.load(X + base + (cols + RD // 2) % RD, cols < D, other=0).to(tl.float32)
        other = tl.where(cols < RD // 2, -other, other)
        cos = tl.load(COS + position * RD + cols, cols < RD, other=1).to(tl.float32)
        sin = tl.load(SIN + position * RD + cols, cols < RD, other=0).to(tl.float32)
        a, b = x * cos, other * sin
        if not NORM and not PRECISE:
            a = a.to(tl.float16).to(tl.float32)
            b = b.to(tl.float16).to(tl.float32)
        if head < H:
            tl.store(Q + (batch * H + head) * D + cols, a + b, cols < D)
        else:
            tl.store(K + ((cache_slot * KH + head - H) * C + cache_position) * D + cols, a + b, cols < D)
            if MATERIALIZE:
                tl.store(TMPK + (batch * KH + head - H) * D + cols, a + b, cols < D)
    else:
        tl.store(V + ((cache_slot * KH + head - H - KH) * C + cache_position) * D + cols, x, cols < D)
        if MATERIALIZE:
            tl.store(TMPV + (batch * KH + head - H - KH) * D + cols, x, cols < D)



def rope_batch(qkv, cos, sin, positions, slots, cache_k, cache_v, heads, q_norm=None, k_norm=None, eps=1e-6, precise=False, page_table=None, materialize=False):
    _, kv_heads, context, dim = cache_k.shape
    rotary_dim = cos.shape[-1]
    if not 0 < rotary_dim <= dim or rotary_dim % 2 or (q_norm is not None and rotary_dim != dim):
        raise ValueError('unsupported rotary table width')
    query = torch.empty((qkv.shape[0], heads, 1, dim), device=qkv.device, dtype=qkv.dtype)
    # Prefill consumes contiguous current-token K/V without gathering cache pages.
    key = torch.empty((qkv.shape[0], kv_heads, dim), device=qkv.device, dtype=cache_k.dtype) if materialize else cache_k
    value = torch.empty_like(key) if materialize else cache_v
    _rope_batch[(qkv.shape[0], heads + 2 * kv_heads)](
        qkv, cos, sin, positions, slots, query, cache_k, cache_v,
        qkv if q_norm is None else q_norm, qkv if k_norm is None else k_norm,
        key, value, materialize,
        slots if page_table is None else page_table, page_table is not None, 0 if page_table is None else page_table.shape[1],
        q_norm is not None, eps, precise, heads, kv_heads, dim, rotary_dim, context, tr.next_power_of_2(dim), enable_fp_fusion=False)
    return (query, key, value) if materialize else query


@tr.jit
def _linear_fp32(X, W, Y, K: tl.constexpr, N: tl.constexpr,
                 BK: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    outputs = tl.program_id(0) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    weight = tl.load(W + outputs[:, None] * K + columns[None, :],
                     (outputs[:, None] < N) & (columns[None, :] < K), other=0).to(tl.float32)
    for row in range(ROWS):
        x = tl.load(X + row * K + columns, columns < K, other=0).to(tl.float32)
        value = tl.sum(weight * x[None, :], 1)
        tl.store(Y + row * N + outputs, value, outputs < N)


def linear_fp32(x, weight):
    """Small-batch FP32 arithmetic directly over FP16/FP32 stored weights."""
    rows, width = x.shape
    if rows > 4:
        return torch.nn.functional.linear(x.float(), weight.float())
    out = torch.empty((rows, weight.shape[0]), device=x.device, dtype=torch.float32)
    _linear_fp32[(tr.cdiv(weight.shape[0], 8),)](
        x, weight, out, width, weight.shape[0], tr.next_power_of_2(width), 8, rows,
        num_warps=4, enable_fp_fusion=False)
    return out
