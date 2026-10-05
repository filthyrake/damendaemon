"""Experimental grouped tensor-core decode; softmax state remains FP32."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _grouped(Q, K, V, POS, SLOTS, OUT, H: tl.constexpr, KH: tl.constexpr,
             D: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    group_id = tl.program_id(0)
    batch, kv_head = group_id // KH, group_id % KH
    rows = tl.arange(0, 16)
    cols = tl.arange(0, D)
    group = H // KH
    heads = kv_head*group + rows
    position = tl.load(POS + batch)
    slot = tl.load(SLOTS + batch)
    q = tl.load(Q + (batch*H + heads[:, None])*D + cols[None, :], rows[:, None] < group, other=0)
    maximum = tl.full((16,), -float('inf'), tl.float32)
    denominator = tl.full((16,), 0., tl.float32)
    accumulator = tl.full((16, D), 0., tl.float32)
    for block in range(tl.cdiv(position+1, BLOCK)):
        times = block*BLOCK + tl.arange(0, BLOCK)
        valid = (times <= position) & (times < C)
        offsets = ((slot*KH + kv_head)*C + times[:, None])*D + cols[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0)
        scores = tl.dot(q, tl.trans(k)) * (D**-.5)
        scores = tl.where(valid[None, :], scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum-next_max)
        weights = tl.exp(scores-next_max[:, None])
        denominator = denominator*correction + tl.sum(weights, 1)
        v = tl.load(V + offsets, valid[:, None], other=0)
        # Preserve the FP32 probabilities as two FP16 pieces. A single rounded
        # product failed the unchanged whole-model batch-layout RMS gate.
        high = weights.to(v.dtype)
        low = (weights-high.to(tl.float32)).to(v.dtype)
        accumulator = accumulator*correction[:, None]
        accumulator = tl.dot(low, v, accumulator)
        accumulator = tl.dot(high, v, accumulator)
        maximum = next_max
    result = accumulator/denominator[:, None]
    tl.store(OUT + (batch*H + heads[:, None])*D + cols[None, :], result, rows[:, None] < group)


def attention(q, k, v, positions, slots, block=64):
    batch, heads, _, dim = q.shape
    _, kv_heads, context, _ = k.shape
    if (heads, kv_heads, dim) != (32, 8, 128) or q.dtype != torch.float16:
        raise ValueError('experiment requires Qwen3-4B FP16 attention geometry')
    out = torch.empty_like(q)
    _grouped[(batch*kv_heads,)](q, k, v, positions, slots, out,
        heads, kv_heads, dim, context, block, num_warps=4, num_stages=1)
    return out
