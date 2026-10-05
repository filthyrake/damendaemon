"""Validation-only selection from real A2 AWQ projection measurements."""
import hashlib
from pathlib import Path
import torch
from dllm import awq_kernels
from . import awq_gemv


def install_validation_dispatch():
    if torch.cuda.get_device_name() != 'NVIDIA A2':
        raise ValueError('AWQ decode candidate is measured only on NVIDIA A2')
    original = awq_kernels.linear
    limits = {(896, 1152): 2, (896, 896): 4, (896, 9728): 3, (4864, 896): 4}

    def linear(x, matrix, bias=None):
        if (x.is_cuda and x.dtype == torch.float16 and matrix.group == 128
                and 1 <= x.shape[0] <= limits.get((matrix.inputs, matrix.outputs), 0)):
            return awq_gemv.linear(x, matrix, bias, 128)
        return original(x, matrix, bias)

    awq_kernels.linear = linear
    return dict(kind='A2 small-row AWQ direct reduction', tile=128,
                scope=[dict(inputs=k[0], outputs=k[1], max_rows=v) for k, v in limits.items()],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                kernel_sha256=hashlib.sha256(Path(awq_gemv.__file__).read_bytes()).hexdigest())
