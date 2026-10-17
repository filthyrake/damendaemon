"""Optional Q4_0 Marlin layout replacement with executor-owned GPU state.

Source coefficients and signed scales are preserved. Marlin rounds the scaled
coefficients to FP16; it is not bitwise identical to residual FP32 scaling.
Device kernels and format attribution live in third_party/marlin.
"""
import ctypes
import os
from dataclasses import dataclass

import torch

from .q4_marlin_policy import LayoutPlan, plan


@dataclass
class Prepared:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None
    inputs: int
    outputs: int

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.weight,self.scales,self.zeros) if t is not None)


class Library:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.damendaemon_q4_marlin_abi.argtypes = []
        self.lib.damendaemon_q4_marlin_abi.restype = integer
        self.abi = self.lib.damendaemon_q4_marlin_abi()
        if self.abi not in (1,2):
            raise ValueError('unsupported native Q4 Marlin ABI')
        self.lib.damendaemon_q4_marlin_init.argtypes = [ctypes.POINTER(integer)]*2
        self.lib.damendaemon_q4_marlin_init.restype = integer
        self.lib.damendaemon_q4_marlin_gemm.argtypes = [ptr]*8+[integer]*5+[ptr]
        self.lib.damendaemon_q4_marlin_gemm.restype = integer
        sms, shared = integer(), integer()
        self.check(self.lib.damendaemon_q4_marlin_init(ctypes.byref(sms),ctypes.byref(shared)))
        self.sms, self.shared = sms.value, shared.value
        if self.sms <= 0 or self.shared <= 0:
            raise ValueError('invalid Q4 Marlin device metadata')
        # Selection does not allocate GPU tensors. Allocation follows admission.
        self.scratch = self.locks = None
        self.device = None

    @property
    def workspace_bytes(self):
        return 4*self.sms*64*256*4 + 4*self.sms*4

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'native Q4 Marlin CUDA error {status}')

    def allocate_workspace(self, device):
        if self.device is not None:
            raise RuntimeError('Q4 Marlin workspace already belongs to an executor')
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.scratch = torch.empty(4*self.sms*64*256,dtype=torch.float32,device=device)
            self.locks = torch.zeros(4*self.sms,dtype=torch.int32,device=device)
        self.device = self.scratch.device

    def prepare(self, weight, scales):
        if (weight.ndim != 2 or not weight.is_cuda or weight.dtype != torch.uint8
                or not weight.is_contiguous() or weight.device != self.device):
            raise ValueError('invalid canonical Q4 Marlin weight')
        n,k = weight.shape[0],weight.shape[1]*2
        if (n <= 0 or k <= 0 or n%128 or k%128 or weight.numel() >= 2**31
                or scales.dtype != torch.float16 or scales.device != weight.device
                or tuple(scales.shape) != (n,k//32) or not scales.is_contiguous()):
            raise ValueError('invalid canonical Q4 Marlin geometry or scales')
        from .q4_marlin_repack import repack
        with torch.cuda.device(weight.device):
            packed = repack(weight)
            order = torch.tensor([i+8*j for i in range(8) for j in range(8)],device=weight.device)
            prepared_scales = scales.T.contiguous().reshape(-1,64)[:,order].reshape(k//32,n).contiguous()
            zeros = (torch.full((k//32,n//8),-2004318072,dtype=torch.int32,device=weight.device)
                     if self.abi == 1 else None)
        return Prepared(packed,prepared_scales,zeros,k,n)

    def linear(self, x, matrix):
        if (x.ndim != 2 or x.dtype != torch.float16 or x.device != self.device
                or x.device != matrix.weight.device or x.shape[1] != matrix.inputs
                or x.shape[0] < 1 or self.scratch is None or self.locks is None):
            raise ValueError('invalid native Q4 Marlin activation or workspace')
        with torch.cuda.device(x.device):
            x = x.contiguous()
            result = torch.empty((x.shape[0],matrix.outputs),dtype=x.dtype,device=x.device)
            self.check(self.lib.damendaemon_q4_marlin_gemm(x.data_ptr(),matrix.weight.data_ptr(),
                matrix.scales.data_ptr(),matrix.zeros.data_ptr() if matrix.zeros is not None else None,None,result.data_ptr(),
                self.scratch.data_ptr(),self.locks.data_ptr(),x.shape[0],matrix.outputs,
                matrix.inputs,self.sms,self.shared,torch.cuda.current_stream(x.device).cuda_stream))
        return result


def select(config, weights, scales, quantization, device, dtype, precision, context,
           max_batch, fused):
    path = os.environ.get('DAMENDAEMON_Q4_MARLIN_LIBRARY')
    if not path or device != 'cuda':
        return None, LayoutPlan()
    selected = plan(config,weights,scales,quantization,device,dtype,precision,context,
                    max_batch,fused,torch.cuda.get_device_name())
    if not selected.names:
        return None, selected
    library = Library(path)
    return library, selected.without_zero_points() if library.abi == 2 else selected


def replace_packing(linear, weights, library, selected):
    """Replace sequentially after transfer aliases are dropped and VRAM admitted."""
    library.allocate_workspace(weights[selected.names[0]].device)
    total = 0
    for name in selected.names:
        weight = weights[name]
        scales = linear.scales.pop(id(weight))
        prepared = library.prepare(weight,scales)
        weights[name] = prepared.weight
        linear.q4_marlin[id(prepared.weight)] = prepared
        total += prepared.bytes
        del weight,scales
    if total != selected.prepared_bytes:
        raise AssertionError('Q4 Marlin prepared layout differs from admitted plan')
