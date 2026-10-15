"""Explicit CUDA packed routed composition with bounded per-route scratch."""
import torch

from .cuda_packed_expert import CudaPackedExpert
from .expert_routes import grouped_routes


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
def cuda_packed_moe(hidden, experts, selected, coefficients, *, shared=None):
    """Add routed contributions in stable expert order, with optional shared FFN.

    IDs/coefficients/gathers remain on CUDA. Range validation synchronizes one
    boolean, then grouping copies one count vector to the host. This is explicit
    host scheduling, not a fused routing kernel. Duplicate routes are additive.
    Only one gathered row and expert result are live at a time. Resident bank,
    output accumulator, route sort/count buffers and allocator overhead require
    separate admission; each expert owns its existing scratch allowance.
    """
    _validate(hidden,experts,selected,coefficients,shared)
    if selected.numel() and bool(((selected < 0) | (selected >= len(experts))).any()):
        # Check before bincount: an invalid huge ID must not allocate huge counts.
        raise ValueError('CUDA route outside prepared bank')
    return _accumulate(hidden,experts,selected,coefficients,shared)


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
