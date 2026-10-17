"""Explicit prepared CUDA expert; no routing, placement or automatic dispatch."""
from dataclasses import dataclass

import torch

from .cuda_packed_projection import cuda_packed_linear
from .expert_ffn import activate_expert
from .packed_projection import projection_plan
from .packed_tiles import validate_layout


@dataclass(frozen=True)
class CudaPackedMatrix:
    weight: torch.Tensor
    scales: torch.Tensor
    format: str

    @property
    def shape(self):
        return validate_layout(self.weight,self.scales,format=self.format,device_type='cuda')


class CudaPackedExpert:
    """Borrow immutable resident matrices, expanding only bounded weight tiles.

    One token row executes at a time to bound live activation scratch. This
    intentionally favors a simple capability foundation over throughput. It is
    not a device routing kernel or hardware FP4/FP8 activation-quantized kernel.
    Resident tensors, result, allocator/BLAS overhead need separate admission.
    """
    def __init__(self,gate,up,down,*,row_chunk=64,column_chunk=256,
                 workspace_bytes=8*1024*1024,activation_limit=None):
        if any(not isinstance(m,CudaPackedMatrix) for m in (gate,up,down)):
            raise ValueError('CUDA expert requires three prepared CUDA matrices')
        intermediate,width=gate.shape
        if up.shape!=(intermediate,width) or down.shape!=(width,intermediate):
            raise ValueError('CUDA expert projection geometry disagrees')
        device=gate.weight.device
        if any(m.weight.device!=device for m in (up,down)):
            raise ValueError('CUDA expert projections must share one device')
        if activation_limit is not None and (type(activation_limit) not in (int,float)
                or not 0<activation_limit<=torch.finfo(torch.float32).max):
            raise ValueError('CUDA expert activation limit must be finite and positive')
        if type(workspace_bytes) is not int or workspace_bytes<1:
            raise ValueError('CUDA expert workspace must be a positive integer')
        projection_bytes=workspace_bytes-128*(width+intermediate)
        for matrix in (gate,up,down):
            projection_plan(matrix.weight,matrix.scales,format=matrix.format,row_chunk=row_chunk,
                column_chunk=column_chunk,workspace_bytes=projection_bytes,device_type='cuda')
        self.gate,self.up,self.down=gate,up,down
        self.width,self.intermediate,self.device=width,intermediate,device
        self.options=dict(row_chunk=row_chunk,column_chunk=column_chunk,workspace_bytes=projection_bytes)
        self.workspace_bytes,self.activation_limit=workspace_bytes,activation_limit

    def _project(self,hidden,matrix):
        return cuda_packed_linear(hidden,matrix.weight,matrix.scales,format=matrix.format,**self.options)

    def _row(self,hidden,coefficient):
        gate=self._project(hidden,self.gate).to(hidden.dtype).float()
        up=self._project(hidden,self.up).to(hidden.dtype).float()
        activated=activate_expert(gate,up,coefficient,self.activation_limit).to(hidden.dtype)
        return self._project(activated,self.down).to(hidden.dtype)

    @torch.no_grad()
    def forward(self,hidden,*,route_weights=None):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=2 or hidden.shape[1]!=self.width
                or hidden.device!=self.device or hidden.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError('CUDA expert input must match device and floating [tokens,width]')
        if route_weights is not None and (not isinstance(route_weights,torch.Tensor)
                or route_weights.shape!=(hidden.shape[0],1) or route_weights.dtype!=torch.float32
                or route_weights.device!=self.device):
            raise ValueError('CUDA route weights must match device and FP32 [tokens,1]')
        output=torch.empty_like(hidden)
        for row in range(hidden.shape[0]):
            coefficient=None if route_weights is None else route_weights[row:row+1]
            output[row:row+1]=self._row(hidden[row:row+1],coefficient)
        return output

    @torch.no_grad()
    def forward_batched(self,hidden,*,route_weights=None):
        """Batched rows through one projection call per layer of the expert.

        Identical per-row arithmetic and cast order as ``_row``; only the
        projection calls see multiple rows at once.
        """
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=2 or hidden.shape[1]!=self.width
                or hidden.device!=self.device or hidden.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError('CUDA expert input must match device and floating [tokens,width]')
        if route_weights is not None and (not isinstance(route_weights,torch.Tensor)
                or route_weights.shape!=(hidden.shape[0],1) or route_weights.dtype!=torch.float32
                or route_weights.device!=self.device):
            raise ValueError('CUDA route weights must match device and FP32 [tokens,1]')
        gate=self._project(hidden,self.gate).to(hidden.dtype).float()
        up=self._project(hidden,self.up).to(hidden.dtype).float()
        activated=activate_expert(gate,up,route_weights,self.activation_limit).to(hidden.dtype)
        return self._project(activated,self.down).to(hidden.dtype)
