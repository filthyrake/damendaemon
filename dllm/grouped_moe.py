"""Device-counted grouped FP32/Q8 expert projections."""
from dataclasses import dataclass
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _count(SELECTED, COUNTS, R: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(SELECTED+rows, rows<R, other=0)
    tl.atomic_add(COUNTS+expert, 1, (rows<R)&(expert>=0)&(expert<E))


@tr.jit
def _offsets(COUNTS, STARTS, TILE_ENDS, E: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    experts = tl.arange(0, BLOCK)
    count = tl.load(COUNTS+experts, experts<E, other=0)
    starts = tl.cumsum(count)-count
    tile_ends = tl.cumsum(tl.cdiv(count, BM))
    tl.store(STARTS+experts, starts, experts<E)
    tl.store(TILE_ENDS+experts, tile_ends, experts<E)


@dataclass
class Routes:
    order: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    tile_ends: torch.Tensor
    experts: int
    top: int
    rows: int
    tiles: int
    block_rows: int


def prepare_routes(selected, experts, block_rows=16):
    # Internal top-k output is required: indices must be in [0, experts).
    if selected.ndim != 2 or selected.dtype != torch.int64 or not selected.is_cuda or not selected.numel():
        raise ValueError('nonempty CUDA int64 routes required')
    if experts < 1 or experts > 256 or block_rows not in (16, 32):
        raise ValueError('unsupported grouped projection geometry')
    selected = selected.contiguous()
    rows = selected.numel()
    order = torch.argsort(selected.reshape(-1), stable=True)
    counts = torch.zeros(experts, device=selected.device, dtype=torch.int32)
    starts, tile_ends = torch.empty_like(counts), torch.empty_like(counts)
    _count[(tr.cdiv(rows, 256),)](selected, counts, rows, experts, 256)
    _offsets[(1,)](counts, starts, tile_ends, experts, block_rows, tr.next_power_of_2(experts))
    # Sum ceil(count/BM) <= ceil(total/BM)+E-1; no host read of counts.
    return Routes(order, counts, starts, tile_ends, experts, selected.shape[1], rows,
                  tr.cdiv(rows, block_rows)+experts-1, block_rows)


@tr.jit
def _linear(X, W, S, ORDER, COUNTS, STARTS, TILE_ENDS, Y,
            E: tl.constexpr, TOP: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            SHARED: tl.constexpr, QUANTIZED: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr):
    tile = tl.program_id(0)
    experts = tl.arange(0, BE)
    ends = tl.load(TILE_ENDS+experts, experts<E, other=2147483647)
    expert = tl.sum((tile>=ends).to(tl.int32), 0)
    if expert<E:
        previous = tl.load(TILE_ENDS+expert-1, expert>0, other=0)
        local_rows = (tile-previous)*BM + tl.arange(0, BM)
        count = tl.load(COUNTS+expert)
        start = tl.load(STARTS+expert)
        route = tl.load(ORDER+start+local_rows, local_rows<count, other=0)
        source = route//TOP if SHARED else route
        outputs = tl.program_id(1)*BN + tl.arange(0, BN)
        columns = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0., tl.float32)
        for step in range(tl.cdiv(K, BK)):
            k = step*BK + columns
            x = tl.load(X+source[:, None]*K+k[None, :],
                        (local_rows[:, None]<count)&(k[None, :]<K), other=0).to(tl.float32)
            values = tl.load(W+(expert*N+outputs[None, :])*K+k[:, None],
                             (outputs[None, :]<N)&(k[:, None]<K), other=0).to(tl.float32)
            if QUANTIZED:
                scales = tl.load(S+(expert*N+outputs[None, :])*(K//32)+(k[:, None]//32),
                                 (outputs[None, :]<N)&(k[:, None]<K), other=0).to(tl.float32)
                values = values*scales
            accumulator = tl.dot(x, values, accumulator, input_precision='tf32x3')
        tl.store(Y+route[:, None]*N+outputs[None, :], accumulator,
                 (local_rows[:, None]<count)&(outputs[None, :]<N))


def linear(x, weight, scales, routes, outputs, shared_input):
    if x.dtype != torch.float32 or not x.is_cuda or x.shape[-1] % 32:
        raise ValueError('grouped projection requires CUDA FP32 inputs aligned to Q8 groups')
    if weight.shape != (routes.experts*outputs, x.shape[-1]):
        raise ValueError('expert matrix dimensions disagree')
    expected_rows = routes.rows//routes.top if shared_input else routes.rows
    if x.shape[0] != expected_rows or weight.device != x.device or routes.order.device != x.device:
        raise ValueError('expert input rows or devices disagree')
    if weight.dtype not in (torch.int8, torch.float16) or not weight.is_contiguous():
        raise ValueError('contiguous Q8 or FP16 expert storage required')
    if (weight.dtype == torch.int8) != (scales is not None):
        raise ValueError('Q8 expert storage requires scales')
    if scales is not None and (scales.shape != (weight.shape[0], x.shape[-1]//32)
            or scales.dtype != torch.float16 or scales.device != x.device or not scales.is_contiguous()):
        raise ValueError('invalid Q8 expert scales')
    x = x.contiguous()
    result = torch.empty((routes.rows, outputs), device=x.device, dtype=x.dtype)
    _linear[(routes.tiles, tr.cdiv(outputs, 64))](x, weight, weight if scales is None else scales,
        routes.order, routes.counts, routes.starts, routes.tile_ends, result,
        routes.experts, routes.top, outputs, x.shape[-1], shared_input, scales is not None,
        routes.block_rows, 64, 64, tr.next_power_of_2(routes.experts), num_warps=4, num_stages=2)
    return result

