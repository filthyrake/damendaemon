"""Independent slot-indexed Gated DeltaNet scan and causal convolution kernels.

State is FP32, laid out [slot, value head, key dimension, value dimension].
Each program owns a disjoint slice of state, including during graph replay.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _scan(Q, K, V, G, B, STATE, OUT, SLOTS,
          T: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
          DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
          RESET: tl.constexpr):
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(SLOTS + row)
    ki = tl.arange(0, BK)
    vi = block * BV + tl.arange(0, BV)
    ptr = STATE + ((slot * HV + head) * DK + ki[:, None]) * DV + vi[None, :]
    mask = (ki[:, None] < DK) & (vi[None, :] < DV)
    if RESET:
        state = tl.full((BK, BV), 0, tl.float32)
    else:
        state = tl.load(ptr, mask, other=0)
    for t in range(T):
        kh = head // (HV // HK)
        q = tl.load(Q + ((row*T+t)*HK+kh)*DK+ki, ki < DK, other=0).to(tl.float32)
        k = tl.load(K + ((row*T+t)*HK+kh)*DK+ki, ki < DK, other=0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q*q, 0) + 1.e-6) * (DK ** -0.5)
        k = k * tl.rsqrt(tl.sum(k*k, 0) + 1.e-6)
        v = tl.load(V + ((row*T+t)*HV+head)*DV+vi, vi < DV, other=0).to(tl.float32)
        decay = tl.exp(tl.load(G + (row*T+t)*HV+head).to(tl.float32))
        beta = tl.load(B + (row*T+t)*HV+head).to(tl.float32)
        state = state * decay
        prediction = tl.sum(state * k[:, None], 0)
        delta = (v - prediction) * beta
        state = state + k[:, None] * delta[None, :]
        result = tl.sum(state * q[:, None], 0)
        tl.store(OUT + ((row*T+t)*HV+head)*DV+vi, result, vi < DV)
    tl.store(ptr, state, mask)


def scan(q, k, v, decay, beta, state, slots, reset=False):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    n, length, kh, kd = q.shape
    vh, vd = v.shape[-2:]
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    _scan[(n, vh, tr.cdiv(vd, 32))](q, k, v, decay.contiguous(), beta.contiguous(),
        state, out, slots, length, kh, vh, kd, vd, tr.next_power_of_2(kd), 32, reset,
        num_warps=4, enable_fp_fusion=False)
    return out


@tr.jit
def _conv(X, W, STATE, SLOTS, Y, D: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    value = tl.load(X + row * D + cols, cols < D, other=0).to(tl.float32)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for i in tl.static_range(C-1):
        previous = tl.load(STATE + (slot*D+cols)*C+i+1, cols < D, other=0).to(tl.float32)
        weight = tl.load(W + cols*C+i, cols < D, other=0).to(tl.float32)
        acc += previous * weight
        tl.store(STATE + (slot*D+cols)*C+i, previous, cols < D)
    weight = tl.load(W + cols*C+C-1, cols < D, other=0).to(tl.float32)
    acc += value * weight
    tl.store(STATE + (slot*D+cols)*C+C-1, value, cols < D)
    tl.store(Y + row*D+cols, acc / (1 + tl.exp(-acc)), cols < D)


def conv_step(x, weight, state, slots):
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _conv[(x.shape[0], tr.cdiv(x.shape[-1], 256))](x, weight, state, slots, out,
        x.shape[-1], weight.shape[-1], 256, enable_fp_fusion=False)
    return out


@tr.jit
def _head(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for block in range(tr.cdiv(K, BK)):
        k = block*BK + inner
        x = tl.load(X + rows[:, None]*K + k[None, :], (rows[:, None]<M)&(k[None, :]<K), other=0)
        w = tl.load(W + cols[None, :]*K + k[:, None], (cols[None, :]<N)&(k[:, None]<K), other=0)
        total = tl.dot(x, w, total)
    tl.store(Y + rows[:, None]*N + cols[None, :], total, (rows[:, None]<M)&(cols[None, :]<N))


def output_head(x, weight):
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
    _head[(tr.cdiv(x.shape[0], 16), tr.cdiv(weight.shape[0], 64))](x, weight, out,
        x.shape[0], weight.shape[0], x.shape[1], 16, 64, 64)
    return out
