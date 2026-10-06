"""Isolated Q4_0-to-Marlin layout adapter; never installed in production dispatch.

Repacking preserves integer coefficients and signed FP16 scales. Prepared
weights are additional resident storage while the canonical Q4 buffers remain.
The separately licensed device kernels retain their existing attribution.
"""
import ctypes
from dataclasses import dataclass

import torch


@dataclass
class Prepared:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None
    inputs: int
    outputs: int

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.weight, self.scales, self.zeros, self.bias) if t is not None)


class Library:
    abi = 1

    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.dllm_q4_marlin_abi.argtypes = []
        self.lib.dllm_q4_marlin_abi.restype = integer
        if self.lib.dllm_q4_marlin_abi() != self.abi:
            raise ValueError('unsupported experimental Q4 Marlin ABI')
        self.lib.dllm_q4_marlin_init.argtypes = [ctypes.POINTER(integer)]*2
        self.lib.dllm_q4_marlin_init.restype = integer
        self.lib.dllm_q4_marlin_repack.argtypes = [ptr]*2+[integer]*4+[ptr]
        self.lib.dllm_q4_marlin_repack.restype = integer
        self.lib.dllm_q4_marlin_gemm.argtypes = [ptr]*8+[integer]*5+[ptr]
        self.lib.dllm_q4_marlin_gemm.restype = integer
        sms, shared = integer(), integer()
        self.check(self.lib.dllm_q4_marlin_init(ctypes.byref(sms), ctypes.byref(shared)))
        self.sms, self.shared = sms.value, shared.value
        # One stream and one sequential execution owner in this experiment.
        self.scratch = torch.empty(4*self.sms*64*256, dtype=torch.float32, device='cuda')
        self.locks = torch.zeros(4*self.sms, dtype=torch.int32, device='cuda')

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'experimental Q4 Marlin CUDA error {status}')

    def prepare(self, weight, scales, bias=None):
        n, k = weight.shape[0], weight.shape[1]*2
        if (not weight.is_cuda or weight.dtype != torch.uint8 or not weight.is_contiguous()
                or n % 128 or k % 128 or scales.shape != (n, k//32)
                or scales.dtype != torch.float16 or scales.device != weight.device):
            raise ValueError('requires aligned canonical CUDA Q4_0 buffers')
        device = weight.device
        blocks = weight.reshape(n, k//32, 16).to(torch.int32)
        coefficients = torch.cat((blocks & 15, blocks >> 4), dim=-1).reshape(n, k)
        channels = coefficients.T.contiguous().reshape(k, n//8, 8)
        shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=device)*4
        awq = (channels << shifts).sum(-1, dtype=torch.int32).contiguous()
        # Independently undo the intermediate layout before the native repack.
        from dllm.awq import unpack_channels
        if not torch.equal(unpack_channels(awq).T, coefficients):
            raise AssertionError('Q4 coefficient permutation changed values')
        del blocks, channels, coefficients
        packed = torch.empty((k//16, n*2), dtype=torch.int32, device=device)
        self.check(self.lib.dllm_q4_marlin_repack(awq.data_ptr(), packed.data_ptr(),
            k, n, self.sms, self.shared, torch.cuda.current_stream(device).cuda_stream))
        permutation = torch.tensor([i+8*j for i in range(8) for j in range(8)], device=device)
        source_scales = scales.T.contiguous()
        prepared_scales = source_scales.reshape(-1, 64)[:, permutation].reshape(k//32, n).contiguous()
        if not torch.equal(prepared_scales.reshape(-1, 64)[:, torch.argsort(permutation)].reshape_as(source_scales), source_scales):
            raise AssertionError('Q4 scale permutation changed values')
        # Every zero point is eight, so the Marlin channel permutation leaves
        # the packed word unchanged. 0x88888888 is negative as signed int32.
        zeros = torch.full((k//32, n//8), -2004318072, dtype=torch.int32, device=device)
        if bias is not None:
            if bias.dtype != torch.float16 or bias.device != device or bias.shape != (n,):
                raise ValueError('invalid Q4 Marlin bias')
            order = torch.tensor([2*i+j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)], device=device)
            bias = bias.reshape(-1, 32)[:, order].reshape(n).contiguous()
        return Prepared(packed, prepared_scales, zeros, bias, k, n)

    def linear(self, x, matrix):
        if (x.ndim != 2 or x.dtype != torch.float16 or x.device != matrix.weight.device
                or x.shape[1] != matrix.inputs or x.shape[0] < 1):
            raise ValueError('invalid experimental Q4 Marlin activation')
        x = x.contiguous()
        result = torch.empty((x.shape[0], matrix.outputs), dtype=x.dtype, device=x.device)
        self.check(self.lib.dllm_q4_marlin_gemm(x.data_ptr(), matrix.weight.data_ptr(),
            matrix.scales.data_ptr(), matrix.zeros.data_ptr() if matrix.zeros is not None else None,
            matrix.bias.data_ptr() if matrix.bias is not None else None,
            result.data_ptr(), self.scratch.data_ptr(), self.locks.data_ptr(),
            x.shape[0], matrix.outputs, matrix.inputs, self.sms, self.shared,
            torch.cuda.current_stream(x.device).cuda_stream))
        return result
