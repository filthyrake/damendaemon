"""Explicit validation access, including formats excluded by production selection."""
import os
import torch
from damendaemon.native_kquant_residual import library, linear as _linear


def linear(x, weight, layout, bias=None):
    return _linear(x, weight, layout, bias,
                   library_path=os.environ['DAMENDAEMON_EXPERIMENTAL_RESIDUAL_LIBRARY'])


def install_validation_dispatch():
    """Install only inside an explicitly selected validation process.

    Restrict to the measured winning A2 shapes. All other projections retain
    production dispatch; this is not a general device autotuning policy.
    """
    import hashlib
    from pathlib import Path
    from damendaemon import native_kquant

    path = Path(os.environ['DAMENDAEMON_EXPERIMENTAL_RESIDUAL_LIBRARY']).resolve()
    library(str(path))  # Fail before running any model checks.
    original = native_kquant.linear_if_enabled
    batch2 = os.environ.get('DAMENDAEMON_EXPERIMENTAL_RESIDUAL_BATCH2') == '1'

    def experimental(x, weight, layout, bias=None):
        width, spec = layout
        single = x.ndim == 2 and x.shape[0] == 1 and width == 2048
        paired = (batch2 and x.ndim == 2 and x.shape[0] == 2
                  and (width, spec[-1][1]) in ((8192, 2048), (2048, 128256))
                  and all(kind == 14 for _, _, kind, _ in spec))
        if ((single or paired) and x.dtype == torch.float16 and x.is_cuda
                and all(kind in (12, 13, 14) for _, _, kind, _ in spec)):
            return linear(x, weight, layout, bias)
        return original(x, weight, layout, bias)

    native_kquant.linear_if_enabled = experimental
    return dict(name='residual-int8-v2', scope='FP16 single-row width-2048 K-quants', batch2_q6_candidate=batch2,
                library=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_backend_label_excludes_experimental_override=True)
