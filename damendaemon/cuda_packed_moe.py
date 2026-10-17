"""Explicit CUDA packed routed composition with bounded per-route scratch."""
import torch

from .cuda_packed_expert import CudaPackedExpert
from .expert_routes import grouped_routes
from .expert_ffn import activate_expert
from .packed_tiles import _decode_tile


def _validate(hidden, experts, selected, coefficients, shared):
    if (not isinstance(experts,(list,tuple)) or not experts
            or any(not isinstance(e,CudaPackedExpert) for e in experts)):
        raise ValueError('CUDA MoE requires a nonempty prepared expert bank')
    device,width = experts[0].device,experts[0].width
    if device.type != 'cuda' or any(e.device != device or e.width != width for e in experts):
        raise ValueError('CUDA expert bank devices/widths disagree')
    if shared is not None and (not isinstance(shared,CudaPackedExpert)
            or shared.device != device or shared.width != width):
        raise ValueError('CUDA shared expert device/width disagrees')
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim != 2 or hidden.shape[1] != width
            or hidden.device != device or hidden.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
        raise ValueError('CUDA MoE needs same-device floating [tokens,width]')
    if (not isinstance(selected,torch.Tensor) or selected.ndim != 2
            or selected.shape[0] != hidden.shape[0] or not 1 <= selected.shape[1] <= len(experts)
            or selected.dtype != torch.int64 or selected.device != device):
        raise ValueError('CUDA routes must be same-device int64 [tokens,top]')
    if (not isinstance(coefficients,torch.Tensor) or coefficients.shape != selected.shape
            or coefficients.device != device or coefficients.dtype != torch.float32):
        raise ValueError('CUDA coefficients must be same-device FP32 matching routes')


@torch.no_grad()
def cuda_packed_moe(hidden, experts, selected, coefficients, *, shared=None, batched=False,
                    fused=False, fused_route_chunk=64, max_gather_bytes=32*1024**2):
    """Add routed contributions in stable expert order, with optional shared FFN.

    IDs/coefficients/gathers remain on CUDA. Range validation synchronizes one
    boolean, then grouping copies one count vector to the host. This is explicit
    host scheduling, not a fused routing kernel. Duplicate routes are additive.
    Only one gathered row and expert result are live at a time. Resident bank,
    output accumulator, route sort/count buffers and allocator overhead require
    separate admission; each expert owns its existing scratch allowance.

    ``batched=True`` keeps the same single count readback but runs one
    multi-row projection call per expert (``forward_batched``) instead of one
    call per token row. Per-row arithmetic and cast order are identical;
    duplicate-route accumulation order is deterministic on CPU and atomic on
    CUDA, so shared-device comparisons use tolerances. Bank entries must expose
    ``forward_batched``.
    """
    _validate(hidden,experts,selected,coefficients,shared)
    if selected.numel() and bool(((selected < 0) | (selected >= len(experts))).any()):
        # Check before bincount: an invalid huge ID must not allocate huge counts.
        raise ValueError('CUDA route outside prepared bank')
    if batched:
        participants = list(experts)+([shared] if shared is not None else [])
        if any(not hasattr(entry,'forward_batched') for entry in participants):
            raise ValueError('batched composition requires experts with forward_batched')
        return _accumulate_batched(hidden,experts,selected,coefficients,shared)
    if fused:
        return _accumulate_fused(hidden,experts,selected,coefficients,shared,
                                 route_chunk=fused_route_chunk,max_gather_bytes=max_gather_bytes)
    return _accumulate(hidden,experts,selected,coefficients,shared)


def _accumulate_batched(hidden,experts,selected,coefficients,shared):
    """Per-expert batched rows; the same single host count readback as _accumulate."""
    output = torch.zeros_like(hidden,dtype=torch.float32)
    flat_coefficients = coefficients.reshape(-1)
    token_ids = torch.arange(hidden.shape[0],device=hidden.device).repeat_interleave(selected.shape[1])
    route_expert = selected.reshape(-1)
    order = torch.argsort(route_expert,stable=True)
    counts = torch.bincount(route_expert,minlength=len(experts)).cpu().tolist()
    if len(counts)!=len(experts):raise ValueError('expert route exceeds configured expert count')
    start = 0
    for expert_id,count in enumerate(counts):
        if count:
            rows = order[start:start+count]
            source = hidden.index_select(0,token_ids[rows])
            coefficient = flat_coefficients.index_select(0,rows).reshape(-1,1)
            result = experts[expert_id].forward_batched(source,route_weights=coefficient)
            output.index_add_(0,token_ids[rows],result.float())
            start += count
    if shared is not None and hidden.shape[0]:
        output.add_(shared.forward_batched(hidden).float())
    return output.to(hidden.dtype)


def _stacked(experts, kind):
    """Stack one packed projection layer across the bank; one format required."""
    formats = {getattr(e, kind).format for e in experts}
    if len(formats) != 1:
        raise ValueError('fused composition requires one packed format across the bank')
    weights = torch.stack([getattr(e, kind).weight for e in experts])
    scales = torch.stack([getattr(e, kind).scales for e in experts])
    return weights, scales, formats.pop()


def _fused_project(hidden_rows, stacked, route_expert, *, rows, columns, row_chunk,
                   column_chunk, route_chunk, max_gather_bytes):
    """Per-route gathered weight tiles with batched bmm; no host readback.

    hidden_rows [c, features] FP32; returns [c, rows]. Tile bounds and route
    chunks are static; the gathered tile [c, tile_rows, tile_cols] is checked
    against an explicit byte budget before allocation.
    """
    weights, scales, format = stacked
    experts_count = weights.shape[0]
    gathered_bytes = route_chunk*min(row_chunk, rows)*min(column_chunk, columns)*4
    if type(max_gather_bytes) is not int or not 1 <= max_gather_bytes <= 2**31 or gathered_bytes > max_gather_bytes:
        raise ValueError('fused gathered-tile workspace budget exceeded')
    output = torch.zeros(hidden_rows.shape[0], rows, dtype=torch.float32, device=hidden_rows.device)
    for row in range(0, rows, row_chunk):
        end_row = min(rows, row+row_chunk)
        for column in range(0, columns, column_chunk):
            end_column = min(columns, column+column_chunk)
            tiles = torch.stack([_decode_tile(weights[e], scales[e], format=format, row_start=row,
                                             row_stop=end_row, column_start=column, column_stop=end_column)
                                 for e in range(experts_count)])
            route_tiles = tiles.index_select(0, route_expert)
            partial = torch.bmm(hidden_rows[:, None, column:end_column], route_tiles.transpose(1, 2))[:, 0, :]
            output[:, row:end_row].add_(partial)
    return output


def _accumulate_fused(hidden, experts, selected, coefficients, shared, *,
                      route_chunk=64, max_gather_bytes=32*1024**2):
    """No host readback: stacked banks, per-route gathered tiles, device accumulation.

    Route order (token-major) differs from the grouped paths' expert-major order,
    so duplicate-route accumulation rounding differs; compare with tolerances.
    """
    gate_stack = _stacked(experts, 'gate'); up_stack = _stacked(experts, 'up'); down_stack = _stacked(experts, 'down')
    gate_rows = gate_stack[0].shape[1]
    hidden_float = hidden.float()
    token_ids = torch.arange(hidden.shape[0], device=hidden.device).repeat_interleave(selected.shape[1])
    route_expert = selected.reshape(-1)
    routes = route_expert.shape[0]
    options = dict(route_chunk=route_chunk, max_gather_bytes=max_gather_bytes)
    limit = experts[0].activation_limit
    gate_columns = gate_stack[0].shape[2]*2 if gate_stack[2] in ('e2m1', 'q4_0_f16') else gate_stack[0].shape[2]
    if hidden_float.shape[1] != gate_columns:
        raise ValueError('fused projection input width disagrees with the stacked bank')
    output = torch.zeros_like(hidden, dtype=torch.float32)
    if routes:
        flat_coefficients = coefficients.reshape(-1)
        route_hidden = hidden_float.index_select(0, token_ids)
        for start in range(0, routes, route_chunk):
            stop = min(routes, start+route_chunk)
            chunk_expert = route_expert[start:stop]
            chunk_rows = route_hidden[start:stop]
            coefficient = flat_coefficients[start:stop].reshape(-1, 1)
            gate = _fused_project(chunk_rows, gate_stack, chunk_expert, rows=gate_stack[0].shape[1],
                                  columns=gate_columns, row_chunk=64, column_chunk=256, **options)
            gate = gate.to(hidden.dtype).float()
            up = _fused_project(chunk_rows, up_stack, chunk_expert, rows=up_stack[0].shape[1],
                                columns=gate_columns, row_chunk=64, column_chunk=256, **options)
            up = up.to(hidden.dtype).float()
            activated = activate_expert(gate, up, coefficient, limit).to(hidden.dtype)
            down_columns = down_stack[0].shape[2]*2 if down_stack[2] in ('e2m1', 'q4_0_f16') else down_stack[0].shape[2]
            result = _fused_project(activated.float(), down_stack, chunk_expert, rows=down_stack[0].shape[1],
                                    columns=down_columns, row_chunk=64, column_chunk=256, **options)
            output.index_add_(0, token_ids[start:stop], result.float())
    if shared is not None and hidden.shape[0]:
        output.add_(shared.forward_batched(hidden).float())
    return output.to(hidden.dtype)


def _accumulate(hidden,experts,selected,coefficients,shared):
    output = torch.zeros_like(hidden,dtype=torch.float32)
    flat_coefficients = coefficients.reshape(-1)
    for expert_id,tokens,positions in grouped_routes(selected,len(experts)):
        expert = experts[expert_id]
        for row in range(tokens.numel()):
            token = tokens[row:row+1]
            coefficient = flat_coefficients.index_select(0,positions[row:row+1]).reshape(1,1)
            source = hidden.index_select(0,token)
            result = expert._row(source,coefficient)
            output.index_add_(0,token,result.float())
            del source,result,coefficient
    if shared is not None:
        for row in range(hidden.shape[0]):
            result = shared._row(hidden[row:row+1],None)
            output[row:row+1].add_(result.float())
            del result
    return output.to(hidden.dtype)
