"""Grouped split attention preserving the existing 256-token softmax partition."""
import torch
import triton as tr
import triton.language as tl
from damendaemon.kernels import _merge


@tr.jit
def _grouped_split(Q, K, V, POS, SLOTS, PART, LSE, H: tl.constexpr,
                   KH: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                   SPLITS: tl.constexpr, GROUP: tl.constexpr):
    group_id, split = tl.program_id(0), tl.program_id(1)
    batch = group_id // (H // GROUP)
    head_start = (group_id % (H // GROUP))*GROUP
    heads = head_start + tl.arange(0, GROUP)
    cols = tl.arange(0, D)
    position = tl.load(POS + batch)
    slot = tl.load(SLOTS + batch)
    offsets_out = ((batch*H + heads[:, None])*SPLITS + split)*D + cols[None, :]
    offsets_lse = (batch*H + heads)*SPLITS + split
    if split*256 <= position:
        times = split*256 + tl.arange(0, 256)
        valid = (times <= position) & (times < C)
        q = tl.load(Q + (batch*H + heads[:, None])*D + cols[None, :]).to(tl.float32)
        kv_head = head_start // (H // KH)
        offsets = ((slot*KH + kv_head)*C + times[:, None])*D + cols[None, :]
        k = tl.load(K + offsets, valid[:, None], other=0).to(tl.float32)
        scores = tl.sum(q[:, None, :]*k[None, :, :], 2)*(D**-.5)
        scores = tl.where(valid[None, :], scores, -float('inf'))
        maximum = tl.max(scores, 1)
        weights = tl.exp(scores-maximum[:, None])
        denominator = tl.sum(weights, 1)
        v = tl.load(V + offsets, valid[:, None], other=0).to(tl.float32)
        result = tl.sum(weights[:, :, None]*v[None, :, :], 1)/denominator[:, None]
        tl.store(PART + offsets_out, result)
        tl.store(LSE + offsets_lse, maximum+tl.log(denominator))
    else:
        tl.store(PART + offsets_out, 0.)
        tl.store(LSE + offsets_lse, -float('inf'))


def attention(q, k, v, positions, slots, group=2):
    batch, heads, _, dim = q.shape
    _, kv_heads, context, _ = k.shape
    if (heads, kv_heads, dim) != (32, 8, 128) or group not in (2, 4):
        raise ValueError('experiment requires Qwen3-4B grouped attention geometry')
    splits = tr.cdiv(context, 256)
    out = torch.empty_like(q)
    partial = torch.empty((batch*heads, splits, dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch*heads, splits), device=q.device, dtype=torch.float32)
    _grouped_split[(batch*heads//group, splits)](q, k, v, positions, slots,
        partial, lse, heads, kv_heads, dim, context, splits, group, num_warps=4*group)
    _merge[(batch*heads,)](partial, lse, out, positions, False,
        heads, dim, splits, tr.next_power_of_2(splits), dim)
    return out
