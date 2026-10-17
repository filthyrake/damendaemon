"""Explicit validation-only global dispatch for the FP32 residual candidate."""
import os
import torch
from damendaemon.native_kquant_residual_f32 import library, linear as _linear


def linear(x, weight, layout, bias=None):
    return _linear(x, weight, layout, bias,
                   library_path=os.environ['DAMENDAEMON_EXPERIMENTAL_RESIDUAL_F32_LIBRARY'])


def install_validation_dispatch():
    import hashlib
    from pathlib import Path
    from damendaemon import native_kquant

    path = Path(os.environ['DAMENDAEMON_EXPERIMENTAL_RESIDUAL_F32_LIBRARY']).resolve()
    library(str(path))
    original = native_kquant.linear_if_enabled

    def experimental(x, weight, layout, bias=None):
        width, spec = layout
        if (x.ndim == 2 and 1 <= x.shape[0] <= 4 and width in (2048, 8192)
                and x.dtype == torch.float32 and x.is_cuda
                and all(kind in (12, 13, 14) for _, _, kind, _ in spec)):
            return linear(x, weight, layout, bias)
        return original(x, weight, layout, bias)

    native_kquant.linear_if_enabled = experimental
    return dict(name='residual-int8-float32', abi=101,
                scope='FP32 rows 1-4; widths 2048/8192; K-quant projections',
                library=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_backend_label_excludes_experimental_override=True)
