"""Explicit, unqualified hybrid Q4 bring-up; separate from dense Q4 support."""
import os

from .device import device_name, selected_accelerator
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
    """Non-chain Q4 context bound: 512, or an explicit bounded ROCm opt-in.

    #68: the guarded 32K cell (golden4096 4K completed under the improved
    thermal envelope) may opt in with DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT
    =<tokens>, valid 512..32768, ROCm/gfx1151 only. Discrete A30 admission is
    unchanged; the chain keeps its own qualified bound.
    """
    if chain:
        return CHAIN_CONTEXT_LIMIT
    value = os.environ.get('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT', '')
    if not value:
        return 512
    if not (value.isascii() and value.isdecimal() and 512 <= int(value) <= 32768):
        raise ValueError('DAMENDAEMON_EXPERIMENTAL_HYBRID_Q4_CONTEXT accepts 512..32768')
    if selected_accelerator() != 'rocm':
        raise ValueError('the experimental Q4 context extension is ROCm gfx1151 only')
    import torch
    architecture = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '')
    if not architecture.startswith('gfx1151'):
        raise ValueError('the experimental Q4 context extension is gfx1151 only')
    return int(value)


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
                and device_name() != 'NVIDIA A30'
                and not _pipeline_pair_anchor())):
        raise ValueError('experimental hybrid Q4 requires explicit opt-in, ROCm or NVIDIA A30 '
                         '(or the explicit distributed-chain opt-in), compensated text execution '
                         f'and context <={context_limit}')
