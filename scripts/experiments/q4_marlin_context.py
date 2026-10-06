"""Explicit longer-context qualification override, not normal runtime support.

Only the Marlin eligibility filter sees the already-qualified context length.
Executor context, rotary tables, KV capacity, admission and actual prompt sizes
retain the requested length. The other model/layout/hardware filters remain.
"""
import hashlib
from pathlib import Path

from dllm import native_q4_marlin


def install_validation_context(limit):
    if limit != 2048:
        raise ValueError('this isolated context experiment is bounded to 2048')
    original = native_q4_marlin.select

    def select(config, weights, scales, quantization, device, dtype, precision,
               context, max_batch, fused):
        qualification_context = 512 if 512 < context <= limit else context
        library, selected = original(config,weights,scales,quantization,device,dtype,
                                     precision,qualification_context,max_batch,fused)
        if qualification_context != context and library is not None and library.abi != 2:
            raise ValueError('long-context experiment requires Q4 Marlin ABI 2')
        return library, selected

    native_q4_marlin.select = select
    return dict(kind='isolated Q4 Marlin context eligibility override',
                maximum_context=limit, qualification_context=512,
                execution_context_unchanged=True, kv_and_admission_context_unchanged=True,
                production_context_support=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
