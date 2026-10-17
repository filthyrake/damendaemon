"""Per-executor opt-in for qualified compensated FP32 K-quant prefill."""
import os

import torch


def linear(x, weight, layout):
    from .kquant_prefill_kernels import linear as execute
    return execute(x, weight, layout, bm=64 if x.shape[0] >= 256 else 32, bn=64)


def select(config, layouts, device, dtype, precision, context):
    if (os.environ.get('DAMENDAEMON_KQUANT_PREFILL_F32') != '1'
            or device != 'cuda' or dtype != torch.float32 or precision != 'float32'
            or not 0 < context <= 2048 or config.get('model_type') != 'llama'
            or config.get('hidden_size') != 2048 or not layouts
            or {kind for _, spec in layouts.values() for _, _, kind, _ in spec}
                not in ({12, 14}, {13, 14})):
        return None

    def prefill(x, weight, layout, bias=None):
        if x.ndim < 2 or x.numel() // x.shape[-1] < 128 or layout[0] not in (2048, 8192):
            return None
        shape = x.shape[:-1]
        output = linear(x.reshape(-1, x.shape[-1]).contiguous(), weight, layout)
        if bias is not None:
            output += bias
        return output.reshape(*shape, layout[1][-1][1])

    return prefill
