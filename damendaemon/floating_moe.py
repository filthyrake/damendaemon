"""Floating shared/routed expert composition with bounded gather workspace.

Component development only: packed weights, distributed ownership and model
admission are separate. Parameters remain in their supplied resident dtype.
"""
import torch

from .expert_ffn import activate_expert, expert_feed_forward


def _shared(shared, hidden):
    if shared is None:
        return 0
    if not isinstance(shared, (tuple, list)) or len(shared) != 3:
        raise ValueError('shared expert must provide gate/up/down projections')
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 or t.dtype != hidden.dtype
           or t.device != hidden.device for t in shared):
        raise ValueError('shared expert projections must match hidden dtype/device')
    gate, up, down = shared
    width, intermediate = hidden.shape[1], gate.shape[0]
    if intermediate < 1 or gate.shape != (intermediate, width) or up.shape != gate.shape or down.shape != (width, intermediate):
        raise ValueError('shared expert projection shapes disagree')
    return intermediate


def _route_chunk(hidden, gate, up, down, selected, coefficients, start, stop, top, limit):
    routes = torch.arange(start, stop, device=hidden.device)
    tokens = routes//top
    experts = selected.reshape(-1)[start:stop].long()
    source = hidden.index_select(0, tokens).unsqueeze(-1)
    gated = torch.bmm(gate.index_select(0, experts), source).squeeze(-1).float()
    raised = torch.bmm(up.index_select(0, experts), source).squeeze(-1).float()
    activated = activate_expert(gated, raised, coefficients.reshape(-1)[start:stop, None], limit)
    result = torch.bmm(down.index_select(0, experts), activated.to(hidden.dtype).unsqueeze(-1)).squeeze(-1)
    return tokens, result.float()


@torch.no_grad()
def floating_moe(hidden, gate, up, down, selected, coefficients, *, shared=None,
                 activation_limit=None, workspace_bytes=64*1024*1024):
    """Compose preselected routed experts plus an unweighted shared expert.

    hidden [tokens,width]; gate/up [experts,intermediate,width]; down
    [experts,width,intermediate]; selected integer [tokens,top]; coefficients
    FP32 [tokens,top]. Selection helpers provide in-range IDs; index_select rejects
    invalid IDs rather than wrapping negative ones. Duplicate routes are additive.

    Workspace budget bounds explicit tensor gathers/intermediates conservatively,
    excluding resident inputs/weights, FP32 output accumulation, returned output
    and backend matmul scratch. This is not a device-memory admission guarantee.
    No full weight promotion or tensor-to-host route extraction occurs. Callers
    retain ownership and must not mutate parameters during execution.
    """
    chunk = validate_moe_inputs(hidden, gate, up, down, selected, coefficients, shared=shared,
                                activation_limit=activation_limit, workspace_bytes=workspace_bytes)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for start in range(0, selected.numel(), chunk):
        tokens, contribution = _route_chunk(hidden, gate, up, down, selected, coefficients,
                                             start, min(start+chunk, selected.numel()), selected.shape[1], activation_limit)
        output.index_add_(0, tokens, contribution)
        # Release returned chunk buffers before allocating the following chunk.
        del tokens, contribution
    if shared is not None:
        for start in range(0, hidden.shape[0], chunk):
            stop = min(start+chunk, hidden.shape[0])
            output[start:stop].add_(expert_feed_forward(hidden[start:stop], *shared,
                                                       activation_limit=activation_limit).float())
    return output.to(hidden.dtype)


def validate_moe_inputs(hidden, gate, up, down, selected, coefficients, *, shared=None,
                        activation_limit=None, workspace_bytes=64*1024*1024):
    """Validate metadata and return route chunk size without executing projections."""
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[1] < 1
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('MoE hidden state must be a FP16/BF16/FP32 matrix')
    if any(not isinstance(t, torch.Tensor) or t.ndim != 3 or t.device != hidden.device
           or t.dtype != hidden.dtype for t in (gate, up, down)):
        raise ValueError('expert banks must be rank-three tensors matching hidden dtype/device')
    experts, intermediate, width = gate.shape
    if experts < 1 or intermediate < 1 or width != hidden.shape[1] or up.shape != gate.shape or down.shape != (experts, width, intermediate):
        raise ValueError('routed expert projection shapes disagree')
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 2 or selected.shape[0] != hidden.shape[0]
            or not 1 <= selected.shape[1] <= experts or selected.dtype not in (torch.int32, torch.int64)
            or selected.device != hidden.device or not selected.is_contiguous()):
        raise ValueError('routes must be contiguous integer [tokens,top] on the expert device')
    if (not isinstance(coefficients, torch.Tensor) or coefficients.shape != selected.shape
            or coefficients.dtype != torch.float32 or coefficients.device != hidden.device or not coefficients.is_contiguous()):
        raise ValueError('route coefficients must be contiguous FP32 matching selected routes')
    if (activation_limit is not None and (type(activation_limit) not in (int, float)
                                         or not 0 < activation_limit <= torch.finfo(torch.float32).max)):
        raise ValueError('expert activation limit must be finite and positive')
    shared_width = _shared(shared, hidden)
    if type(workspace_bytes) is not int or workspace_bytes < 1:
        raise ValueError('expert workspace budget must be a positive integer')
    # Three gathered projections plus generous activation/index temporary space.
    per_route = 3*intermediate*width*hidden.element_size()+64*max(intermediate, shared_width)+32*width+128
    chunk = workspace_bytes//per_route
    if chunk < 1:
        raise ValueError('expert workspace budget cannot fit one route')
    return chunk
