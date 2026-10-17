"""Explicit CUDA packed projection adapter; no automatic model dispatch."""
import itertools

import torch

from .packed_projection import projection_plan, validate_row_range
from .cuda_packed_projection import cuda_packed_linear


class CudaPackedProjection:
    """Borrow immutable raw weights/scales with an explicit result storage dtype.

    Row views retain original matrix coordinates, including E8M0/FP32 block scales or canonical Q4/Q8 FP16 row scales.
    This is an explicit CUDA adapter; output casting does not emulate activation
    quantization or accelerator accumulation. No dense weight is retained.
    """
    def __init__(self, weight, scales, *, format, dtype=torch.float32, row_range=None,
                 row_chunk=64, column_chunk=256, workspace_bytes=8*1024*1024):
        if dtype not in (torch.float16,torch.bfloat16,torch.float32):
            raise ValueError('packed projection result dtype must be floating')
        rows,columns,_ = projection_plan(weight,scales,format=format,row_chunk=row_chunk,
                                         column_chunk=column_chunk,workspace_bytes=workspace_bytes,device_type='cuda')
        self.start,self.stop = validate_row_range(row_range,rows)
        self.weight,self.scales,self.format = weight,scales,format
        self.shape,self.ndim = (self.stop-self.start,columns),2
        self.dtype,self.device = dtype,weight.device
        self.options = dict(row_chunk=row_chunk,column_chunk=column_chunk,workspace_bytes=workspace_bytes)

    def rows(self,start,stop):
        start,stop = validate_row_range((start,stop),self.shape[0])
        return CudaPackedProjection(self.weight,self.scales,format=self.format,dtype=self.dtype,
                                row_range=(self.start+start,self.start+stop),**self.options)

    @torch.no_grad()
    def linear(self,hidden):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim < 2 or hidden.shape[-1] != self.shape[1]
                or hidden.dtype != self.dtype or hidden.device != self.device):
            raise ValueError('packed projection input dtype/device/shape mismatch')
        output = torch.empty((*hidden.shape[:-1],self.shape[0]),dtype=self.dtype,device=self.device)
        # Visit leading batches without flattening/copying a strided whole input.
        for index in itertools.product(*(range(n) for n in hidden.shape[:-2])):
            output[index] = cuda_packed_linear(hidden[index],self.weight,self.scales,format=self.format,
                                          row_range=(self.start,self.stop),**self.options).to(self.dtype)
        return output

