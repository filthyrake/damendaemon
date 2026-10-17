"""Isolated exact-coefficient K-quant expansion tile experiment."""
import hashlib
from pathlib import Path

import torch
import triton as tr

from damendaemon import kquant_kernels


def install_validation_dispatch():
    original = kquant_kernels.expand

    def expand(weight, layout, dtype):
        width, spec = layout
        if (dtype != torch.float16 or weight.device.type != 'cuda'
                or width not in (2048, 5632)
                or any(kind not in (12, 14) for _, _, kind, _ in spec)):
            return original(weight, layout, dtype)
        outputs = spec[-1][1]
        result = torch.empty((outputs, width), device=weight.device, dtype=dtype)
        kquant_kernels._expand[(tr.cdiv(outputs, 8), tr.cdiv(width, 512))](
            weight, result, outputs, width, spec, 8, 512,
            num_warps=4, enable_fp_fusion=False)
        return result

    kquant_kernels.expand = expand
    return dict(kind='experimental K-quant expansion tiles', br=8, bc=512, warps=4,
                coefficient_policy='unchanged FP16; CUDA Q4_K/Q6_K widths 2048/5632',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
