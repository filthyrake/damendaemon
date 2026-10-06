"""Direct canonical Q4_0 byte permutation into the attributed Marlin layout.

The output layout follows third_party/marlin/awq_repack.cuh (Apache-2.0,
contributors to vLLM). This independent Triton implementation reads canonical
Q4 bytes directly; it does not materialize an intermediate integer matrix.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _repack(W, Y, N:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    word = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    tile = word//128
    lane = (word%128)//4
    warp = word%4
    output = (tile%(N//64))*64 + warp*16 + lane//4
    column = (tile//(N//64))*16 + (lane%4)*2
    result = tl.full((BLOCK,), 0, tl.uint32)
    for position in tl.static_range(8):
        n = output + ((position//2)%2)*8
        k = column + (position%2)*8 + position//4
        packed = tl.load(W+n*(K//2)+(k//32)*16+k%16, word<N*K//8, other=0).to(tl.uint32)
        value = (packed >> tl.where(k%32<16, 0, 4)) & 15
        result |= value << (position*4)
    tl.store(Y+word, result.to(tl.int32), word<N*K//8)


def repack(weight, block=256):
    if (weight.ndim!=2 or weight.dtype!=torch.uint8 or not weight.is_cuda
            or not weight.is_contiguous() or weight.shape[0]%128 or weight.shape[1]%64
            or min(weight.shape)<=0 or weight.numel()>=2**31
            or block not in (256,512,1024)):
        raise ValueError('requires aligned canonical CUDA Q4 bytes and a supported tile')
    n, k = weight.shape[0], weight.shape[1]*2
    result = torch.empty((k//16,n*2), dtype=torch.int32, device=weight.device)
    _repack[(tr.cdiv(n*k//8,block),)](weight,result,n,k,block)
    return result
