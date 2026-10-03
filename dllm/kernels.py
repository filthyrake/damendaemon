"""Small inference kernels. Matmuls remain with cuBLAS; reductions use FP32.

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
                NORM: tl.constexpr, EPS: tl.constexpr,
                T: tl.constexpr, H: tl.constexpr, KH: tl.constexpr,
                D: tl.constexpr, C: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, D)
    position = tl.load(POS + token)
    base = token * (H + 2 * KH) * D + head * D
    x = tl.load(X + base + cols).to(tl.float32)
    if head < H + KH:
        if NORM:
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            weight = tl.load((QW if head < H else KW) + cols).to(tl.float32)
            x = x * inv * weight
            other_cols = (cols + D // 2) % D
            other = tl.load(X + base + other_cols).to(tl.float32)
            other_weight = tl.load((QW if head < H else KW) + other_cols).to(tl.float32)
            other = other * inv * other_weight
        else:
            other = tl.load(X + base + (cols + D // 2) % D).to(tl.float32)
        other = tl.where(cols < D // 2, -other, other)
        cos = tl.load(COS + position * D + cols).to(tl.float32)
        sin = tl.load(SIN + position * D + cols).to(tl.float32)
        a, b = x * cos, other * sin
        if not NORM:
            a = a.to(tl.float16).to(tl.float32)
            b = b.to(tl.float16).to(tl.float32)
        rotated = a + b
        if head < H:
            tl.store(Q + (head * T + token) * D + cols, rotated)
        else:
            tl.store(K + ((head - H) * C + position) * D + cols, rotated)
    else:
        tl.store(V + ((head - H - KH) * C + position) * D + cols, x)


def rope_store(qkv, cos, sin, positions, cache_k, cache_v, heads, q_norm=None, k_norm=None, eps=1e-6):
    tokens = qkv.shape[0]
    _, kv_heads, context, dim = cache_k.shape
    query = torch.empty((1, heads, tokens, dim), dtype=qkv.dtype, device=qkv.device)
    _rope_store[(tokens, heads + 2 * kv_heads)](
        qkv, cos, sin, positions, query, cache_k, cache_v,
        qkv if q_norm is None else q_norm, qkv if k_norm is None else k_norm,
        q_norm is not None, eps, tokens, heads, kv_heads, dim, context, enable_fp_fusion=False)
    return query


@tr.jit
def _attention(Q, K, V, POS, SLOTS, OUT, PART, LSE,
               H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
               C: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr, SLOTTED: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    position = tl.load(POS + batch)
    slot = tl.load(SLOTS + batch) if SLOTTED else 0
    cols = tl.arange(0, D)
    if split * BLOCK <= position:
        times = split * BLOCK + tl.arange(0, BLOCK)
        valid = (times <= position) & (times < C)
        q = tl.load(Q + bh * D + cols).to(tl.float32)
        kh = slot * KH + head // (H // KH)
        offsets = (kh * C + times[:, None]) * D + cols[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0).to(tl.float32)
        logits = tl.sum(k * q[None, :], 1) * (D ** -0.5)
        logits = tl.where(valid, logits, -float("inf"))
        maximum = tl.max(logits, 0)
        weights = tl.exp(logits - maximum)
        denominator = tl.sum(weights, 0)
        v = tl.load(V + offsets, valid[:, None], other=0).to(tl.float32)
        result = tl.sum(v * weights[:, None], 0) / denominator
        if SPLITS == 1:
            tl.store(OUT + bh * D + cols, result)
        else:
            tl.store(PART + (bh * SPLITS + split) * D + cols, result)
            tl.store(LSE + bh * SPLITS + split, maximum + tl.log(denominator))
    else:
        if SPLITS > 1:
            tl.store(PART + (bh * SPLITS + split) * D + cols, 0.)
            tl.store(LSE + bh * SPLITS + split, -float("inf"))


@tr.jit
def _merge(PART, LSE, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    splits = tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    lse = tl.load(LSE + head * S + splits, splits < S, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, 0))
    weights = weights / tl.sum(weights, 0)
    partial = tl.load(PART + (head * S + splits[:, None]) * D + cols[None, :],
                      splits[:, None] < S, other=0)
    tl.store(OUT + head * D + cols, tl.sum(partial * weights[:, None], 0))


def attention(query, cache_k, cache_v, positions, slots=None):
    batch, heads, _, dim = query.shape
    _, kv_heads, context, _ = cache_k.shape
    block = min(256, tr.next_power_of_2(context))
    splits = tr.cdiv(context, block)
    out = torch.empty_like(query)
    partial = torch.empty((batch * heads, splits, dim), device=query.device, dtype=torch.float32)
    lse = torch.empty((batch * heads, splits), device=query.device, dtype=torch.float32)
    _attention[(batch * heads, splits)](query, cache_k, cache_v, positions,
                               positions if slots is None else slots, out, partial, lse,
                               heads, kv_heads, dim, context, splits, block, slots is not None, num_warps=4)
    if splits > 1:
        _merge[(batch * heads,)](partial, lse, out, dim, splits, tr.next_power_of_2(splits))
    return out


@tr.jit
def _rope_batch(X, COS, SIN, POS, SLOTS, Q, K, V, QW, KW,
                NORM: tl.constexpr, EPS: tl.constexpr,
                H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, C: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, D)
    position, slot = tl.load(POS + batch), tl.load(SLOTS + batch)
    base = batch * (H + 2 * KH) * D + head * D
    x = tl.load(X + base + cols).to(tl.float32)
    if head < H + KH:
        if NORM:
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            weight = tl.load((QW if head < H else KW) + cols).to(tl.float32)
            x = x * inv * weight
            other_cols = (cols + D // 2) % D
            other = tl.load(X + base + other_cols).to(tl.float32)
            other_weight = tl.load((QW if head < H else KW) + other_cols).to(tl.float32)
            other = other * inv * other_weight
        else:
            other = tl.load(X + base + (cols + D // 2) % D).to(tl.float32)
        other = tl.where(cols < D // 2, -other, other)
        cos = tl.load(COS + position * D + cols).to(tl.float32)
        sin = tl.load(SIN + position * D + cols).to(tl.float32)
        a, b = x * cos, other * sin
        if not NORM:
            a = a.to(tl.float16).to(tl.float32)
            b = b.to(tl.float16).to(tl.float32)
        if head < H:
            tl.store(Q + (batch * H + head) * D + cols, a + b)
        else:
            tl.store(K + ((slot * KH + head - H) * C + position) * D + cols, a + b)
    else:
        tl.store(V + ((slot * KH + head - H - KH) * C + position) * D + cols, x)


def rope_batch(qkv, cos, sin, positions, slots, cache_k, cache_v, heads, q_norm=None, k_norm=None, eps=1e-6):
    _, kv_heads, context, dim = cache_k.shape
    query = torch.empty((qkv.shape[0], heads, 1, dim), device=qkv.device, dtype=qkv.dtype)
    _rope_batch[(qkv.shape[0], heads + 2 * kv_heads)](
        qkv, cos, sin, positions, slots, query, cache_k, cache_v,
        qkv if q_norm is None else q_norm, qkv if k_norm is None else k_norm,
        q_norm is not None, eps, heads, kv_heads, dim, context, enable_fp_fusion=False)
    return query
