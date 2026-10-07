"""Prepared packed experts for bounded CPU reference execution.

Routes are read on the CPU intentionally. This control implementation is not a
GPU routing kernel, and does not emulate accelerator activation quantization.
"""
from dataclasses import dataclass

import torch

from .expert_ffn import activate_expert
from .packed_projection import packed_linear, projection_plan
from .packed_tiles import validate_layout


@dataclass(frozen=True)
class PackedMatrix:
    weight: torch.Tensor
    scales: torch.Tensor
    format: str

    @property
    def shape(self):
        return validate_layout(self.weight, self.scales, format=self.format)


class PackedExpert:
    """Borrow three immutable packed projections; expand only individual tiles."""
    def __init__(self, gate, up, down, *, row_chunk=64, column_chunk=256,
                 workspace_bytes=8*1024*1024, activation_limit=None):
        if any(not isinstance(p,PackedMatrix) for p in (gate,up,down)):
            raise ValueError('expert needs three prepared packed matrices')
        intermediate,width = gate.shape
        if up.shape != (intermediate,width) or down.shape != (width,intermediate):
            raise ValueError('packed expert projection geometry disagrees')
        if activation_limit is not None and (type(activation_limit) not in (int,float)
                or not 0 < activation_limit <= torch.finfo(torch.float32).max):
            raise ValueError('packed expert activation limit must be finite and positive')
        if type(workspace_bytes) is not int or workspace_bytes < 1:
            raise ValueError('packed expert workspace must be a positive integer')
        # Reserve live expert outputs, activation/cast temporaries and row input,
        # then give the remaining allowance to one projection at a time.
        projection_bytes = workspace_bytes-128*(width+intermediate)
        for matrix in (gate,up,down):
            projection_plan(matrix.weight,matrix.scales,format=matrix.format,row_chunk=row_chunk,
                            column_chunk=column_chunk,workspace_bytes=projection_bytes)
        self.gate,self.up,self.down = gate,up,down
        self.width,self.intermediate = width,intermediate
        self.options = dict(row_chunk=row_chunk,column_chunk=column_chunk,workspace_bytes=projection_bytes)
        self.workspace_bytes = workspace_bytes
        self.activation_limit = activation_limit

    def _project(self, hidden, matrix):
        return packed_linear(hidden,matrix.weight,matrix.scales,format=matrix.format,**self.options)

    def _row(self, hidden, coefficient):
        # Match storage boundaries of the floating control. FP32 GEMM internals
        # still differ from low-precision accelerator kernels and packed GEMMs.
        gate = self._project(hidden,self.gate).to(hidden.dtype).float()
        up = self._project(hidden,self.up).to(hidden.dtype).float()
        activated = activate_expert(gate,up,coefficient,self.activation_limit).to(hidden.dtype)
        return self._project(activated,self.down).to(hidden.dtype)

    @torch.no_grad()
    def forward(self, hidden, *, route_weights=None):
        _hidden(hidden,self.width)
        if route_weights is not None and (not isinstance(route_weights,torch.Tensor)
                or route_weights.shape != (hidden.shape[0],1) or route_weights.dtype != torch.float32
                or route_weights.device.type != 'cpu'):
            raise ValueError('packed expert route weights must be CPU FP32 [tokens,1]')
        output = torch.empty_like(hidden)
        for token in range(hidden.shape[0]):
            coefficient = None if route_weights is None else route_weights[token:token+1]
            output[token:token+1] = self._row(hidden[token:token+1],coefficient)
        return output


def _hidden(hidden,width):
    if (not isinstance(hidden,torch.Tensor) or hidden.ndim != 2 or hidden.shape[1] != width
            or hidden.dtype not in (torch.float32,torch.float16,torch.bfloat16) or hidden.device.type != 'cpu'):
        raise ValueError('packed expert input must be CPU floating [tokens,width]')


@torch.no_grad()
def packed_moe(hidden, experts, selected, coefficients, *, shared=None):
    """CPU reference routed sum with an optional unweighted shared contribution.

    Duplicate routes are additive. Validate every route ID before computation.
    One token/expert runs at a time, with prepared per-expert scratch allowances.
    Resident weights/inputs, final accumulation/result and BLAS scratch are
    excluded. This is deliberately separate from device-resident GPU routing.
    """
    if (not isinstance(experts,(tuple,list)) or not experts
            or any(not isinstance(e,PackedExpert) for e in experts)):
        raise ValueError('packed MoE needs a nonempty prepared expert sequence')
    width = experts[0].width
    _hidden(hidden,width)
    if any(e.width != width for e in experts) or (shared is not None and
            (not isinstance(shared,PackedExpert) or shared.width != width)):
        raise ValueError('packed routed/shared expert widths disagree')
    if (not isinstance(selected,torch.Tensor) or selected.ndim != 2
            or selected.shape[0] != hidden.shape[0] or not 1 <= selected.shape[1] <= len(experts)
            or selected.dtype not in (torch.int32,torch.int64) or selected.device.type != 'cpu'):
        raise ValueError('packed routes must be CPU integer [tokens,top]')
    if (not isinstance(coefficients,torch.Tensor) or coefficients.shape != selected.shape
            or coefficients.dtype != torch.float32 or coefficients.device.type != 'cpu'):
        raise ValueError('packed coefficients must be CPU FP32 matching routes')
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= len(experts)):
        raise ValueError('packed route ID outside prepared expert bank')
    output = torch.zeros_like(hidden,dtype=torch.float32)
    for token in range(hidden.shape[0]):
        source = hidden[token:token+1]
        for route in range(selected.shape[1]):
            expert = experts[int(selected[token,route])]
            result = expert._row(source,coefficients[token:token+1,route:route+1])
            output[token:token+1].add_(result.float())
            del result
        if shared is not None:
            result = shared._row(source,None)
            output[token:token+1].add_(result.float())
            del result
    return output.to(hidden.dtype)
