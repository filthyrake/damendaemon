"""Experimental grouped online-softmax decode for fixed Qwen3 caches."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _stream(Q, K, V, POS, SLOTS, OUT, H: tl.constexpr, KH: tl.constexpr,
            D: tl.constexpr, C: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    group_id = tl.program_id(0)
    groups_per_batch = H // GROUP
    batch = group_id // groups_per_batch
    head_start = (group_id % groups_per_batch) * GROUP
    heads = head_start + tl.arange(0, GROUP)
    cols = tl.arange(0, D)
    position = tl.load(POS + batch)
    slot = tl.load(SLOTS + batch)
    q = tl.load(Q + (batch*H + heads[:, None])*D + cols[None, :]).to(tl.float32)
    kv_head = head_start // (H // KH)
    maximum = tl.full((GROUP,), -float('inf'), tl.float32)
    denominator = tl.full((GROUP,), 0., tl.float32)
    accumulator = tl.full((GROUP, D), 0., tl.float32)
    for block in range(tl.cdiv(position+1, BLOCK)):
        times = block*BLOCK + tl.arange(0, BLOCK)
        valid = (times <= position) & (times < C)
        offsets = ((slot*KH + kv_head)*C + times[:, None])*D + cols[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0).to(tl.float32)
        scores = tl.sum(q[:, None, :]*k[None, :, :], 2) * (D**-.5)
        scores = tl.where(valid[None, :], scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(maximum-next_max)
        weights = tl.exp(scores-next_max[:, None])
        denominator = denominator*correction + tl.sum(weights, 1)
        v = tl.load(V + offsets, valid[:, None], other=0).to(tl.float32)
        accumulator = accumulator*correction[:, None] + tl.sum(weights[:, :, None]*v[None, :, :], 1)
        maximum = next_max
    result = accumulator/denominator[:, None]
    tl.store(OUT + (batch*H + heads[:, None])*D + cols[None, :], result)


def attention(q, k, v, positions, slots, block=32, group=4):
    batch, heads, _, dim = q.shape
    _, kv_heads, context, _ = k.shape
    if (heads, kv_heads, dim) != (32, 8, 128) or group not in (1, 4):
        raise ValueError('experiment requires Qwen3-4B attention geometry')
    out = torch.empty_like(q)
    _stream[(batch*heads//group,)](q, k, v, positions, slots, out,
        heads, kv_heads, dim, context, group, block, num_warps=4)
    return out
