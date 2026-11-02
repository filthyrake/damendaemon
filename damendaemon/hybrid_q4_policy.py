"""Explicit, unqualified hybrid Q4 bring-up; separate from dense Q4 support."""
import os

from .device import selected_accelerator
from .vision_policy import enabled as vision_enabled


def _pipeline_pair_anchor():
    """True when a quantized two-device pipeline is explicitly opted in.

    #536 gate 2: under DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE=1 with the
    quant opt-in, stages build on each guarded pipeline device inside
    `torch.cuda.device(stage_device)`, so the single-device A30 identity pin
    cannot be evaluated with the current-device name. The anchor moves to the
    PRIMARY pipeline device (cuda:0) while the pipeline policy owns the pair,
    budgets, precision and context admission.
    """
    if selected_accelerator() != 'cuda':
        return False
    from .hybrid_pipeline_policy import budgets, quant_pipeline_enabled
    limits = budgets()
    if limits is None or len(limits) < 2 or not quant_pipeline_enabled():
        return False
    import torch
    return torch.cuda.get_device_name(0) == 'NVIDIA A30'


CHAIN_CONTEXT_LIMIT = 2048


def chain_anchor():
    """True when an explicitly opted-in distributed chain rank requests Q4.

    #536 gate 4: the three-device chain (A30 + A2 + a remote tail rank, e.g.
    the desktop RTX 3090) is Q4-qualified end to end — 12/12 greedy tokens
    byte-equal the A30+A2 in-process reference of the same quantized path at
    context 2048, batch 1 (docs/evidence/distributed/qwen38-gate4-plan-536/
    gate4-27b-three-device-qualified.md). The served path over the same
    topology is qualified by the chain-serve evidence in the same directory.

    The chain opt-in admits exactly that qualified shape: a non-A30 tail
    rank (the A30-primary pair anchor cannot hold on a single-GPU remote
    host) and the chain context window (up to CHAIN_CONTEXT_LIMIT, the
    qualified cell). Every other guard — explicit Q4 opt-in, accelerator,
    qwen3_5_text architecture, compensated precision, no vision — still
    applies, and single-device production admission is unchanged.
    """
    if os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CHAIN', '0') != '1':
        return False
    if selected_accelerator() not in ('cuda', 'rocm'):
        return False
    from .hybrid_pipeline_policy import budgets, quant_pipeline_enabled
    return budgets() is not None and quant_pipeline_enabled()


def _experimental_context_limit(chain):
    """Non-chain Q4 context bound: 512, or an explicit bounded opt-in.

    #68: the guarded 32K cell (golden4096 4K completed under the improved
    thermal envelope) may opt in with DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT
    =<tokens>, valid 512..32768. The extension is structural: it admits on
    any device already admitted by validate() — the A6000 carries the full
    32K evidence set (C1/C2 capacity, lifecycle and cancellation cells,
    docs/evidence/nvidia/a6000-q4-32k-68/), and the previous gfx1151 arch
    string was a name lock, not a capability check (it already exceeded
    that host's own 4K qualification). The chain keeps its own bound.
    """
    if chain:
        return CHAIN_CONTEXT_LIMIT
    value = os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT', '')
    if not value:
        return 512
    if not (value.isascii() and value.isdecimal() and 512 <= int(value) <= 32768):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT accepts 512..32768')
    if selected_accelerator() not in ('rocm', 'cuda'):
        raise ValueError('the experimental Q4 context extension requires an admitted ROCm or CUDA device')
    return int(value)


def _cuda_q4_admits():
    """True when this CUDA host can execute the compensated Q4 hybrid path.

    Structural admission: the fused Triton hybrid kernels import and a
    real CUDA device is present. The retired device-name pins ('NVIDIA
    A30', the gfx1151 arch string) were proxies for 'hardware we had
    measured' — the A6000 now carries the same qualification (C1/C2
    capacity, lifecycle and cancellation cells, all passing), and a device
    without the kernels fails here with a clear refusal instead of at
    first kernel compile. The two-device pipeline keeps its own identity
    anchor (#536 gate 2).
    """
    if _pipeline_pair_anchor():
        return True
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        from . import hybrid_kernels  # noqa: F401 — admission probe
    except ImportError:
        return False
    return True


def validate(config, max_context=None):
    accelerator = selected_accelerator()
    chain = chain_anchor()
    context_limit = _experimental_context_limit(chain)
    if (os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4', '0') != '1'
            or accelerator not in ('rocm', 'cuda')
            or config.get('model_type') != 'qwen3_5_text'
            or os.environ.get('DAMENDAEMON_HYBRID_PRECISION') != 'compensated'
            or (max_context is not None and not 1 <= max_context <= context_limit)
            or vision_enabled()
            or (accelerator == 'cuda' and not chain
                and not _cuda_q4_admits())):
        raise ValueError('experimental hybrid Q4 requires explicit opt-in, ROCm or a '
                         'CUDA device with the Triton hybrid kernels (or the explicit '
                         'distributed-chain opt-in), compensated text execution '
                         f'and context <={context_limit}')
