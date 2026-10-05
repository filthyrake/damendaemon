"""Validation-only, per-executor expanded projection cache with a hard byte cap.

Packed weights remain authoritative. This prototype deliberately has no normal
runtime option: admission accounting must include the additional residency
before production use. Cache tensors belong to Linear, not a global registry.
"""
import hashlib
from pathlib import Path

import torch

CACHE_LIMIT = 3 * 1024**3


def install_validation_dispatch():
    from dllm.quantization import Linear
    from dllm.kquant_kernels import expand

    original_init = Linear.__init__
    original_call = Linear.__call__

    def initialize(self, weights, host_scales, device, fallback, packed_formats=None,
                   packed_decode=None, packed_prefill=None, q4_decode=None):
        original_init(self, weights, host_scales, device, fallback, packed_formats,
                      packed_decode, packed_prefill, q4_decode)
        self.experimental_dense_cache = {}
        self.experimental_dense_cache_bytes = 0
        if device != 'cuda':
            return
        selected = {}
        for name, layout in (packed_formats or {}).items():
            width, spec = layout
            if (name != 'embed' and width in (2048, 5632)
                    and all(kind in (12, 14) for _, _, kind, _ in spec)):
                selected[id(weights[name])] = (weights[name], layout)
        required = sum(2 * layout[0] * layout[1][-1][1] for _, layout in selected.values())
        if required > CACHE_LIMIT:
            return
        for key, (weight, layout) in selected.items():
            self.experimental_dense_cache[key] = expand(weight, layout, torch.float16)
        self.experimental_dense_cache_bytes = required

    def multiply(self, x, weight, bias=None):
        cached = self.experimental_dense_cache.get(id(weight))
        if cached is not None and x.dtype == torch.float16 and x.numel() // x.shape[-1] > 1:
            return torch.nn.functional.linear(x, cached, bias)
        return original_call(self, x, weight, bias)

    Linear.__init__ = initialize
    Linear.__call__ = multiply
    return dict(kind='experimental resident FP16 projection cache', byte_cap=CACHE_LIMIT,
                scope='CUDA Q4_K/Q6_K widths 2048/5632; rows greater than one',
                production_admission_accounts_for_cache=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
