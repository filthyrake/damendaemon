"""#645 gate-2 scoping: single-bf16-dot packed prefill kernel vs the
production compensated kernel — the only residency-feasible candidate.

Dense bf16 weights cannot fit the qualified 32K residency (27B bf16 alone
~54 GiB > the 52.74 GiB Q4-32K admission), so the only candidate shape is
the packed kernel with the 3-dot compensation dropped to a single bf16 dot
(keeping on-the-fly dequant and the Triton tiles). This runner measures
that candidate's arithmetic rate and its FP32-dequant deviation at the
gate-1 shapes, same arms re-measured for a same-epoch comparison.

The kernel below is a single-dot copy of damendaemon.compensated_q8._project
(experiment-local; the production kernel is untouched).
"""
if not __debug__:
    raise RuntimeError('prefill_single_dot_gap_645 asserts under -O; run without -O')

import json
import time

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl

SHAPES = [
    ("0.8b chunk127 mlp", 127, 1024, 3584),
    ("0.8b prefill512 mlp", 512, 1024, 3584),
    ("27b chunk127 mlp", 127, 5120, 17408),
    ("27b prefill512 mlp", 512, 5120, 17408),
]


@tr.jit
def _project_single(X, W, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, Q4: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    groups = tl.arange(0, BK//32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tr.cdiv(K, BK)):
        k = block*BK + inner
        x = tl.load(X + rows[:, None]*K + k[None, :], (rows[:, None]<M)&(k[None, :]<K), other=0)
        if Q4:
            packed = tl.load(W + cols[None, :]*(K//2) + (k[:, None]//32)*16 + k[:, None]%16,
                             (cols[None, :]<N)&(k[:, None]<K), other=0).to(tl.int32)
            shift = tl.where(k[:, None]%32 < 16, 0, 4)
            q = (((packed >> shift) & 15)-8).to(tl.float32)
        else:
            q = tl.load(W + cols[None, :]*K + k[:, None], (cols[None, :]<N)&(k[:, None]<K), other=0).to(tl.float32)
        g = block*(BK//32) + groups
        scale = tl.load(S + cols[None, :]*(K//32) + g[:, None],
                       (cols[None, :]<N)&(g[:, None]<K//32), other=0).to(tl.float32)
        w = (q.reshape(BK//32, 32, BN)*scale[:, None, :]).reshape(BK, BN)
        # Single bf16 dot: no compensation. Strictly less precise than the
        # production 3-dot path by construction.
        acc = tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16), acc)
    tl.store(Y + rows[:, None]*N + cols[None, :], acc, (rows[:, None]<M)&(cols[None, :]<N))


def _single_dot(x, weight, scales, q4):
    storage_dtype = torch.uint8 if q4 else torch.int8
    width = weight.shape[1] * (2 if q4 else 1)
    x = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    _project_single[(tr.cdiv(x.shape[0], 32), tr.cdiv(weight.shape[0], 64))](
        x, weight, scales, out, x.shape[0], weight.shape[0], x.shape[1], 32, 64, 32, q4)
    return out


def pack_q8(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-128, 127, (n, k), generator=g)
    scales = (torch.rand(n, k // 32, generator=g) * 0.02 + 0.001).to(torch.float16)
    return q.to(torch.int8).cuda(), scales.cuda()


def pack_q4(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-8, 8, (n, k), generator=g)
    nib = (q + 8).to(torch.uint8)
    low = nib.reshape(n, k // 32, 2, 16)[:, :, 0, :]
    high = nib.reshape(n, k // 32, 2, 16)[:, :, 1, :]
    packed = (low | (high << 4)).reshape(n, k // 2).contiguous()
    scales = (torch.rand(n, k // 32, generator=g) * 0.02 + 0.001).to(torch.float16)
    return packed.cuda(), scales.cuda()


def dequant_q8(weight, scales):
    w = weight.to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(32, dim=1)
    return w * s


def dequant_q4(packed, scales):
    n = packed.shape[0]
    k = packed.shape[1] * 2
    b = packed.to(torch.int32)
    low = (b & 15) - 8
    high = ((b >> 4) & 15) - 8
    q = torch.cat((low.reshape(n, k // 32, 16), high.reshape(n, k // 32, 16)), dim=2).reshape(n, k)
    s = scales.to(torch.float32).repeat_interleave(32, dim=1)
    return (q * s).contiguous()


def bench(fn, iters, warmup=20):
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
    result = dict(gate='#645 gate-2 scoping: single-dot packed kernel',
                 gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                 shapes=[])
    for label, m, k, n in SHAPES:
        g = torch.Generator().manual_seed(1234)
        x32 = torch.randn(m, k, generator=g).to('cuda', torch.float32)
        wq8, s8 = pack_q8(n, k, 77)
        wq4, s4 = pack_q4(n, k, 78)
        wb16 = torch.randn(n, k, generator=g).cuda().to(torch.bfloat16)
        ref8 = x32 @ dequant_q8(wq8, s8).t()
        ref4 = x32 @ dequant_q4(wq4, s4).t()
        row: dict = dict(shape=label, m=m, k=k, n=n)
        big = 2 * m * k * n
        iters = max(20, min(300, int(3e12 // big)))
        for arm, fn in (
            ('q8_compensated', lambda: linear(x32, wq8, s8)),
            ('q4_compensated', lambda: linear_q4(x32, wq4, s4)),
            ('q8_single_dot', lambda: _single_dot(x32, wq8, s8, False)),
            ('q4_single_dot', lambda: _single_dot(x32, wq4, s4, True)),
            ('dense_bf16', lambda: F.linear(x32.to(torch.bfloat16), wb16)),
        ):
            seconds = bench(fn, iters)
            tops = 2 * m * k * n * iters / seconds / 1e12
            row[arm] = dict(tops=round(tops, 3), iters=iters,
                            ms_per_call=round(seconds / iters * 1e3, 3))
            print(label, arm, row[arm], flush=True)
        for arm, out, ref in (('q8_single_dot', _single_dot(x32, wq8, s8, False), ref8),
                              ('q4_single_dot', _single_dot(x32, wq4, s4, True), ref4)):
            deviation = (out - ref).abs().max().item()
            row[arm]['ref_max_abs'] = round(deviation, 6)
            row[arm]['ref_scale'] = round(ref.abs().max().item(), 3)
            # Single-dot bounds guard against layout bugs only (those produce
            # O(output-scale*sqrt(K)) deviations); its genuine uncompensated
            # bf16 error ~2-3% relative is the gate-2 finding, recorded as
            # measured. Compensated arms keep the tight 0.5 bound.
            assert deviation < 5.0, f'{arm} deviates implausibly (layout bug?): {deviation}'
        result['shapes'].append(row)
        del x32, wq8, s8, wq4, s4, wb16, ref8, ref4
        torch.cuda.empty_cache()
    with open('/tmp/d645b/single-dot-gap.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))

if __name__ == '__main__':
    main()
