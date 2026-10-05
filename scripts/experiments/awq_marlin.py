"""Isolated standalone Marlin C ABI experiment, not runtime registration."""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import torch
from dllm.awq import unpack_channels


@dataclass
class Prepared:
    weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bias: torch.Tensor | None
    scratch: torch.Tensor
    locks: torch.Tensor
    inputs: int
    outputs: int


class Library:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lib = ctypes.CDLL(str(self.path))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.dllm_awq_marlin_abi.argtypes = []
        self.lib.dllm_awq_marlin_abi.restype = integer
        if self.lib.dllm_awq_marlin_abi() != 2:
            raise ValueError('unsupported Marlin experiment ABI')
        self.lib.dllm_awq_marlin_init.argtypes = [ctypes.POINTER(integer)]*2
        self.lib.dllm_awq_marlin_init.restype = integer
        self.lib.dllm_awq_marlin_repack.argtypes = [ptr]*2+[integer]*4+[ptr]
        self.lib.dllm_awq_marlin_repack.restype = integer
        self.lib.dllm_awq_marlin_gemm.argtypes = [ptr]*8+[integer]*5+[ptr]
        self.lib.dllm_awq_marlin_gemm.restype = integer
        sms, shared = integer(), integer()
        self.check(self.lib.dllm_awq_marlin_init(ctypes.byref(sms), ctypes.byref(shared)))
        self.sms, self.shared = sms.value, shared.value

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'Marlin experiment CUDA error {status}')

    def prepare(self, matrix, bias=None, workspace=None):
        if (not matrix.weight.is_cuda or matrix.group != 128 or matrix.inputs % 128
                or matrix.outputs % 128 or matrix.inputs <= 128):
            raise ValueError('Marlin experiment requires CUDA AWQ group 128 with aligned dimensions')
        n, k, device = matrix.outputs, matrix.inputs, matrix.weight.device
        if bias is not None and (bias.dtype != torch.float16 or bias.device != device or tuple(bias.shape) != (n,)):
            raise ValueError('invalid experiment bias')
        packed = torch.empty((k//16, n*2), dtype=torch.int32, device=device)
        self.check(self.lib.dllm_awq_marlin_repack(matrix.weight.data_ptr(), packed.data_ptr(),
            k, n, self.sms, self.shared, torch.cuda.current_stream(device).cuda_stream))
        # Layout permutations follow the pinned upstream Marlin format. Keep
        # the canonical AWQ zero-point decoder independent of the repacker.
        permutation = [i+8*j for i in range(8) for j in range(8)]
        scales = matrix.scales.reshape(-1, 64)[:, permutation].reshape(-1, n).contiguous()
        zeros = unpack_channels(matrix.zeros).reshape(-1, 64)[:, permutation]
        zeros = zeros.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        shifts = torch.arange(8, device=device, dtype=torch.int32)*4
        zeros = (zeros << shifts).sum(-1, dtype=torch.int32).reshape(k//128, n//8).contiguous()
        if bias is not None:
            permutation = [2*i+j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
            bias = bias.reshape(-1, 32)[:, permutation].reshape(n).contiguous()
        # Conservative experiment workspace; runtime admission is a later gate.
        if workspace is None:
            scratch = torch.empty(4*self.sms*64*256, dtype=torch.float32, device=device)
            locks = torch.zeros(4*self.sms, dtype=torch.int32, device=device)
        else:
            scratch, locks = workspace
        return Prepared(packed, scales, zeros, bias, scratch, locks, k, n)

    def linear(self, x, matrix):
        if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.float16
                or x.device != matrix.weight.device or x.shape[1] != matrix.inputs
                or x.shape[0] < 1):
            raise ValueError('Marlin experiment requires nonempty contiguous FP16 rows')
        x = x.contiguous()
        result = torch.empty((x.shape[0], matrix.outputs), device=x.device, dtype=x.dtype)
        self.check(self.lib.dllm_awq_marlin_gemm(x.data_ptr(), matrix.weight.data_ptr(),
            matrix.scales.data_ptr(), matrix.zeros.data_ptr(), None if matrix.bias is None else matrix.bias.data_ptr(),
            result.data_ptr(), matrix.scratch.data_ptr(), matrix.locks.data_ptr(),
            x.shape[0], matrix.outputs, matrix.inputs, self.sms, self.shared,
            torch.cuda.current_stream(x.device).cuda_stream))
        return result
