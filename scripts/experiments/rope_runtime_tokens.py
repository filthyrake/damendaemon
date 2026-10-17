"""Explicit rotary stride specialization experiment; arithmetic is unchanged."""
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["T"])
def rope_store_runtime_tokens(X, COS, SIN, POS, Q, K, V, QW, KW,
                NORM: tl.constexpr, EPS: tl.constexpr, PRECISE: tl.constexpr,
                T, H: tl.constexpr, KH: tl.constexpr,
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
            x = x * inv
            if not PRECISE:
                x = x.to(tl.float16).to(tl.float32)
            x = x * weight
            if not PRECISE:
                x = x.to(tl.float16).to(tl.float32)
            other_cols = (cols + D // 2) % D
            other = tl.load(X + base + other_cols, cols < D, other=0).to(tl.float32)
            other_weight = tl.load((QW if head < H else KW) + other_cols, cols < D, other=0).to(tl.float32)
            other = other * inv
            if not PRECISE:
                other = other.to(tl.float16).to(tl.float32)
            other = other * other_weight
            if not PRECISE:
                other = other.to(tl.float16).to(tl.float32)
        else:
            other = tl.load(X + base + (cols + D // 2) % D, cols < D, other=0).to(tl.float32)
        other = tl.where(cols < D // 2, -other, other)
        cos = tl.load(COS + position * D + cols, cols < D, other=0).to(tl.float32)
        sin = tl.load(SIN + position * D + cols, cols < D, other=0).to(tl.float32)
        a, b = x * cos, other * sin
        if not PRECISE:
            a = a.to(tl.float16).to(tl.float32)
            b = b.to(tl.float16).to(tl.float32)
        rotated = a + b
        if head < H:
            tl.store(Q + (head * T + token) * D + cols, rotated, cols < D)
        else:
            tl.store(K + ((head - H) * C + position) * D + cols, rotated, cols < D)
    else:
        tl.store(V + ((head - H - KH) * C + position) * D + cols, x, cols < D)



def install_validation_dispatch():
    import os
    # Keep this historical candidate independent of the integrated stage policy.
    os.environ['DAMENDAEMON_Q4_STAGE_RUNTIME_ROPE']='0'
    from damendaemon import kernels
    kernels._rope_store=rope_store_runtime_tokens
