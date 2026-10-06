"""Experimental FP32 small-batch Q8 matvec with bounded reduction tiles."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _project(X, W, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.arange(0, BR)
    outputs = tl.program_id(0)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    groups = tl.arange(0, BK//32)
    accumulator = tl.full((BR, BN), 0, tl.float32)
    for block in range(tr.cdiv(K, BK)):
        k = block*BK + inner
        g = block*(BK//32) + groups
        x = tl.load(X + rows[:, None]*K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        q = tl.load(W + outputs[:, None]*K + k[None, :],
                    (outputs[:, None] < N) & (k[None, :] < K), other=0).to(tl.float32)
        scales = tl.load(S + outputs[:, None]*(K//32) + g[None, :],
                         (outputs[:, None] < N) & (g[None, :] < K//32), other=0).to(tl.float32)
        weight = (q.reshape(BN, BK//32, 32)*scales[:, :, None]).reshape(BN, BK)
        accumulator += tl.sum(x[:, None, :]*weight[None, :, :], 2)
    tl.store(Y + rows[:, None]*N + outputs[None, :], accumulator,
             (rows[:, None] < M) & (outputs[None, :] < N))


def linear(x, weight, scales):
    if (x.ndim < 1 or weight.ndim != 2 or scales.ndim != 2 or x.dtype != torch.float32
            or weight.dtype != torch.int8 or scales.dtype != torch.float16):
        raise ValueError('requires FP32 activations and Q8 matrices with FP16 scales')
    width = x.shape[-1]
    if (width <= 0 or width % 32 or weight.shape[1] != width or weight.shape[0] <= 0
            or scales.shape != (weight.shape[0], width//32)):
        raise ValueError('incompatible Q8 dimensions')
    if (x.device.type != 'cuda' or weight.device != x.device or scales.device != x.device
            or not weight.is_contiguous() or not scales.is_contiguous()):
        raise ValueError('requires same-device GPU tensors and contiguous Q8 storage')
    shape = x.shape[:-1]
    x = x.reshape(-1, width).contiguous()
    if not 1 <= x.shape[0] <= 4:
        raise ValueError('blocked matvec is limited to one through four rows')
    output = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32)
    _project[(tr.cdiv(weight.shape[0], 4),)](
        x, weight, scales, output, x.shape[0], weight.shape[0], width,
        tr.next_power_of_2(x.shape[0]), 4, 512, num_warps=4, enable_fp_fusion=False)
    return output.reshape(*shape, weight.shape[0])
