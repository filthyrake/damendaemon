"""Numerical and cache-boundary checks for the fused CUDA kernels."""
import json

import torch
from torch.nn import functional as F

from dllm import kernels
from dllm.runtime import require_gpu


def main():
    require_gpu()
    torch.manual_seed(2026)
    checks = 0
    for width in (896, 1536, 4864):
        for rows in (1, 7):
            x = torch.randn((rows, width), dtype=torch.float16, device="cuda")
            w = torch.randn(width, dtype=torch.float16, device="cuda")
            xf = x.float()
            expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).half() * w
            torch.testing.assert_close(kernels.rms(x, w, 1e-6), expected, atol=0.004, rtol=0.002)
            gate_up = torch.cat((x, x * 0.3), dim=-1)
            torch.testing.assert_close(kernels.swiglu(gate_up), F.silu(x) * (x * 0.3), atol=0.002, rtol=0.002)
            checks += 2
            precise_x = xf + .000123
            precise_norm = precise_x * torch.rsqrt(precise_x.square().mean(-1, keepdim=True)+1e-6)
            torch.testing.assert_close(kernels.rms(precise_x, w, 1e-6, True),
                                       (precise_norm*w.float()).half(), atol=.004, rtol=.002)
            gate, up = gate_up.chunk(2, -1)
            torch.testing.assert_close(kernels.swiglu(gate_up, True),
                                       (F.silu(gate.float())*up.float()).half(), atol=.002, rtol=.002)
            checks += 2
    attention_checks = []
    for dim in (64, 96, 128):
        heads, kv_heads = 14, 2
        for context in (256, 513, 1024):
            k = torch.randn((1, kv_heads, context, dim), dtype=torch.float16, device="cuda")
            v = torch.randn_like(k)
            q = torch.randn((1, heads, 1, dim), dtype=torch.float16, device="cuda")
            for position in sorted({0, 1, 31, min(255, context-1), min(256, context-1), context-1}):
                pos = torch.tensor([position], dtype=torch.long, device="cuda")
                actual = kernels.attention(q, k, v, pos)
                expected = F.scaled_dot_product_attention(q, k[:, :, :position+1], v[:, :, :position+1], enable_gqa=True)
                torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.01)
                attention_checks.append({"dim": dim, "context": context, "position": position,
                                         "max_error": (actual-expected).abs().max().item()})
                checks += 1
            # Cache entries beyond the live prefix must never influence output.
            pos.zero_()
            k[:, :, 1:] = float("nan")
            v[:, :, 1:] = float("nan")
            actual = kernels.attention(q, k, v, pos)
            expected = v[:, :, :1].repeat_interleave(heads // kv_heads, dim=1)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            checks += 1
    # Slotted decode must isolate reordered sequences across attention splits.
    context, dim, heads, kv_heads = 1024, 64, 14, 2
    k = torch.randn((5, kv_heads, context, dim), dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)
    q = torch.randn((4, heads, 1, dim), dtype=torch.float16, device="cuda")
    slots = torch.tensor([4, 1, 3, 0], device="cuda")
    positions = torch.tensor([0, 256, 512, 1023], device="cuda")
    for slot, position in zip(slots.tolist(), positions.tolist()):
        k[slot, :, position+1:] = float("nan")
        v[slot, :, position+1:] = float("nan")
    actual = kernels.attention(q, k, v, positions, slots)
    for row, (slot, position) in enumerate(zip(slots.tolist(), positions.tolist())):
        expected = F.scaled_dot_product_attention(q[row:row+1], k[slot:slot+1, :, :position+1],
                                                  v[slot:slot+1, :, :position+1], enable_gqa=True)
        torch.testing.assert_close(actual[row:row+1], expected, atol=.003, rtol=.01)
        checks += 1
    # Local attention must skip fully expired splits and never read their NaNs.
    for dtype in (torch.float16, torch.float32):
        for window in (1, 127, 256, 512):
            dim, heads, context = 256, 4, 2048
            k = torch.randn((4, 1, context, dim), device='cuda', dtype=dtype)
            v = torch.randn_like(k)
            q = torch.randn((4, heads, 1, dim), device='cuda', dtype=dtype)
            slots = torch.tensor([3, 0, 2, 1], device='cuda')
            positions = torch.tensor([0, 511, 512, 2047], device='cuda')
            expected = []
            for row, (slot, position) in enumerate(zip(slots.tolist(), positions.tolist())):
                start = max(0, position-window+1)
                expected.append(F.scaled_dot_product_attention(q[row:row+1],
                    k[slot:slot+1, :, start:position+1], v[slot:slot+1, :, start:position+1],
                    enable_gqa=True, scale=.07))
                k[slot, :, :start] = v[slot, :, :start] = float('nan')
                k[slot, :, position+1:] = v[slot, :, position+1:] = float('nan')
            actual = kernels.attention(q, k, v, positions, slots, window=window, scale=.07)
            torch.testing.assert_close(actual, torch.cat(expected), atol=.003, rtol=.01)
            checks += 1
    # Qwen3 normalizes each Q/K head before RoPE; values must remain untouched.
    for dim in (64, 96, 128):
        batch, heads, kv_heads, context = 4, 16, 8, 1024
        positions = torch.tensor([0, 255, 256, 1023], device="cuda")
        slots = torch.tensor([3, 1, 2, 0], device="cuda")
        qkv = torch.randn((batch, (heads+2*kv_heads)*dim), dtype=torch.float16, device="cuda")
        cos = torch.randn((context, dim), dtype=torch.float16, device="cuda")
        sin = torch.randn_like(cos)
        for normalized in (False, True):
            qw = torch.randn(dim, dtype=torch.float16, device="cuda") if normalized else None
            kw = torch.randn_like(qw) if normalized else None
            q, k, v = qkv.split([heads*dim, kv_heads*dim, kv_heads*dim], -1)
            def transform(x, weight):
                x = x.reshape(batch, -1, dim)
                if weight is not None:
                    xf = x.float()
                    x = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True)+1e-6) * weight.float()
                left, right = x.chunk(2, -1)
                return (x*cos[positions, None] + torch.cat((-right,left),-1)*sin[positions,None]).half()
            expected_q, expected_k = transform(q,qw), transform(k,kw)
            cache_k = torch.zeros((4,kv_heads,context,dim),dtype=torch.float16,device="cuda")
            cache_v = torch.zeros_like(cache_k)
            actual = kernels.rope_batch(qkv,cos,sin,positions,slots,cache_k,cache_v,heads,qw,kw)
            torch.testing.assert_close(actual[:,:,0],expected_q,atol=.004,rtol=.002)
            torch.testing.assert_close(cache_k[slots,:,positions,:],expected_k,atol=.004,rtol=.002)
            torch.testing.assert_close(cache_v[slots,:,positions,:],v.reshape(batch,kv_heads,dim),atol=0,rtol=0)
            checks += 1
            cache_k.zero_(); cache_v.zero_()
            actual = kernels.rope_store(qkv,cos,sin,positions,cache_k[:1],cache_v[:1],heads,qw,kw)
            torch.testing.assert_close(actual[0].transpose(0,1),expected_q,atol=.004,rtol=.002)
            torch.testing.assert_close(cache_k[0,:,positions,:].transpose(0,1),expected_k,atol=.004,rtol=.002)
            checks += 1
    print(json.dumps({"checks": checks, "attention": attention_checks, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
