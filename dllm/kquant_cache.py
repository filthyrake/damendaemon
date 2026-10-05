"""Opt-in, admission-accounted resident projections for qualified TinyLlama."""
from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class CachePlan:
    names: tuple = ()
    bytes: int = 0


def plan(config, weights, layouts, device, dtype, precision, context, available_bytes):
    raw = os.environ.get('DLLM_KQUANT_DENSE_CACHE_MIB', '0')
    try:
        budget = int(raw) * 1024**2
    except ValueError as error:
        raise ValueError('DLLM_KQUANT_DENSE_CACHE_MIB must be a nonnegative integer') from error
    if budget < 0:
        raise ValueError('DLLM_KQUANT_DENSE_CACHE_MIB must be a nonnegative integer')
    expected = dict(model_type='llama', hidden_size=2048, intermediate_size=5632,
                    num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4,
                    head_dim=64, vocab_size=32000, max_position_embeddings=2048)
    if (not budget or device != 'cuda' or dtype != torch.float16 or precision != 'mixed'
            or not 0 < context <= 2048 or any(config.get(key) != value for key, value in expected.items())
            or config.get('rope_scaling') is not None or config.get('rope_theta', 10000) != 10000
            or config.get('sliding_window') or not layouts
            or {kind for _, spec in layouts.values() for _, _, kind, _ in spec} != {12, 14}):
        return CachePlan()
    names, seen, needed = [], set(), 0
    for name, (width, spec) in layouts.items():
        if name == 'embed' or id(weights[name]) in seen:
            continue
        if width not in (2048, 5632):
            return CachePlan()
        seen.add(id(weights[name]))
        names.append(name)
        needed += 2 * width * spec[-1][1]
    # All-or-nothing selection avoids unqualified partial-cache performance
    # policies. Core weights/KV/workspace have already reserved their memory.
    if needed > min(budget, max(0, available_bytes)):
        return CachePlan()
    return CachePlan(tuple(names), needed)


def build(cache_plan, weights, layouts):
    if not cache_plan.names:
        return {}
    from .kquant_kernels import expand
    return {id(weights[name]): expand(weights[name], layouts[name], torch.float16)
            for name in cache_plan.names}
