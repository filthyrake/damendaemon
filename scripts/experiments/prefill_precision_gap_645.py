"""#645 gate 1: measure the real prefill GEMM gap on HOMER — the production
compensated Triton Q8/Q4 packed projection kernel vs dense FP32 and BF16
references at the decoder prefill shapes (including the production chunk
row count 127).

Arms per shape:
  - q8_compensated : damendaemon.compensated_q8.linear (Q8 packed, 3 BF16 dots)
  - q4_compensated : damendaemon.compensated_q8.linear_q4 (Q4 packed)
  - dense_f32      : torch F.linear, FP32 weights
  - dense_bf16     : torch F.linear, BF16 weights

Each quantized arm also checks its output against the FP32 dequantized-
weight reference (the compensated path is an approximation by construction;
the deviation is recorded as a sanity number, not a pass gate).

Correctness notes: synthetic weights only (this measures kernel arithmetic
rate, not checkpoint quality — admission scope for the compensated
executor path is unchanged and remains what hybrid_precision.select
enforces). Ops counted as 2*M*K*N for every arm.

Run on HOMER (gfx1151) with the production stack. Bounded: minutes of GEMM
load, prefill-path thermally safe at default power (#68 revised envelope).
"""
if not __debug__:
    raise RuntimeError('prefill_precision_gap_645 asserts under -O; run without -O')

import json
import time

import torch
import torch.nn.functional as F

SHAPES = [
    # (label, M, K, N) — decoder prefill shapes incl. the production chunk
    ("0.8b chunk127 mlp", 127, 1024, 3584),
    ("0.8b prefill512 mlp", 512, 1024, 3584),
    ("27b chunk127 mlp", 127, 5120, 17408),
    ("27b prefill512 mlp", 512, 5120, 17408),
]

def pack_q8(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-128, 127, (n, k), generator=g)
    scales = (torch.rand(n, k // 32, generator=g) * 0.02 + 0.001).to(torch.float16)
    return q.to(torch.int8), scales

def dequant_q8(weight, scales):
    n, k = weight.shape
    w = weight.to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(32, dim=1)
    return w * s

def pack_q4(n, k, seed):
    assert k % 32 == 0
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-8, 8, (n, k), generator=g)
    nib = (q + 8).to(torch.uint8)
    # Kernel layout: byte (k//32)*16 + k%16 holds k%32<16 in the low nibble
    # and k%32>=16 (offset 16) in the high nibble.
    low = nib[:, :].reshape(n, k // 32, 2, 16)[:, :, 0, :]      # k%32 in 0..15
    high = nib.reshape(n, k // 32, 2, 16)[:, :, 1, :]           # k%32 in 16..31
    packed = (low | (high << 4)).reshape(n, k // 2).contiguous()
    scales = (torch.rand(n, k // 32, generator=g) * 0.02 + 0.001).to(torch.float16)
    return packed, scales

def dequant_q4(packed, scales):
    n = packed.shape[0]
    k = packed.shape[1] * 2
    b = packed.to(torch.int32)
    low = (b & 15) - 8                     # k%32 in 0..15
    high = ((b >> 4) & 15) - 8             # k%32 in 16..31
    # Within each 32-wide group the layout is 16 low-nibble values then 16
    # high-nibble values (kernel byte (k//32)*16 + k%16).
    q = torch.cat((low.reshape(n, k // 32, 16), high.reshape(n, k // 32, 16)),
                  dim=2).reshape(n, k)
    s = scales.to(torch.float32).repeat_interleave(32, dim=1)
    return (q * s).contiguous()

def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start

def main():
    from damendaemon.compensated_q8 import linear, linear_q4
    torch.cuda.init()
    result = dict(gate='#645 gate 1 prefill GEMM gap',
                 gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                 shapes=[])
    for label, m, k, n in SHAPES:
        g = torch.Generator().manual_seed(1234)
        x32 = torch.randn(m, k, generator=g).to('cuda', torch.float32)
        wq8, s8 = pack_q8(n, k, 77)
        wq4, s4 = pack_q4(n, k, 78)
        wq8, s8 = wq8.cuda(), s8.cuda()
        wq4, s4 = wq4.cuda(), s4.cuda()
        w32 = torch.randn(n, k, generator=g).cuda()
        wb16 = w32.to(torch.bfloat16)
        row: dict = dict(shape=label, m=m, k=k, n=n)
        # FP32 dequantized references for the quantized sanity checks.
        ref8 = x32 @ dequant_q8(wq8, s8).t()
        ref4 = x32 @ dequant_q4(wq4, s4).t()
        # Scale iters with problem size so every arm runs O(seconds).
        big = 2 * m * k * n
        iters = max(20, min(300, int(3e12 // big)))
        for arm, fn in (
            ('q8_compensated', lambda: linear(x32, wq8, s8)),
            ('q4_compensated', lambda: linear_q4(x32, wq4, s4)),
            ('dense_f32', lambda: F.linear(x32, w32)),
            ('dense_bf16', lambda: F.linear(x32.to(torch.bfloat16), wb16)),
        ):
            seconds = bench(fn, iters, 20)
            tops = 2 * m * k * n * iters / seconds / 1e12
            row[arm] = dict(tops=round(tops, 3), iters=iters,
                            ms_per_call=round(seconds / iters * 1e3, 3))
            print(label, arm, row[arm], flush=True)
        # Sanity: the compensated arms approximate the FP32 dequant reference.
        out8 = linear(x32, wq8, s8)
        out4 = linear_q4(x32, wq4, s4)
        for arm, out, ref in (('q8_compensated', out8, ref8), ('q4_compensated', out4, ref4)):
            deviation = (out - ref).abs().max().item()
            row[arm]['ref_max_abs'] = round(deviation, 6)
            assert deviation < 0.5, f'{arm} deviates from the FP32 dequant reference by {deviation}'
        result['shapes'].append(row)
        del x32, wq8, s8, wq4, s4, w32, wb16, ref8, ref4, out8, out4
        torch.cuda.empty_cache()
    with open('/tmp/d645/prefill-gap.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))

if __name__ == '__main__':
    main()
