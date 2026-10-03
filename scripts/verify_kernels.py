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
    attention_checks = []
    for dim in (64, 128):
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
    print(json.dumps({"checks": checks, "attention": attention_checks, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
