"""Request-local random streams and device-side probability filtering."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0
    top_p: float = 1
    top_k: int = 0
    seed: int | None = None

    def __post_init__(self):
        for name in ('temperature', 'top_p'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError('temperature must be nonnegative and top_p must be in (0, 1]')
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError('seed must be an integer in [0, 2**64)')


class Sampler:
    def __init__(self, options):
        self.options = options
        self.constraint = None
        # Never mutate the global RNG, or consume another request's stream.
        self.random = random.Random(options.seed) if options.temperature else None

    def select(self, logits):
        import torch
        options = self.options
        if self.constraint is not None:
            logits = self.constraint.apply(logits)
        if not options.temperature:
            return logits.argmax(-1)
        values = logits.float()
        indices = None
        if options.top_k:
            values, indices = values.topk(min(options.top_k, values.numel()))
        if options.top_p < 1:
            values, order = values.sort(descending=True)
            indices = order if indices is None else indices[order]
        # Subtract the maximum before dividing so small temperatures stay stable.
        probabilities = torch.softmax((values - values.max()) / max(options.temperature, torch.finfo(torch.float32).tiny), -1)
        if options.top_p < 1:
            # Keep the first token that crosses the cumulative probability bound.
            remove = probabilities.cumsum(-1) - probabilities >= options.top_p
            probabilities = probabilities.masked_fill(remove, 0)
            probabilities = probabilities / probabilities.sum()
        if not torch.isfinite(probabilities).all().item():
            raise ValueError('cannot sample non-finite model probabilities')
        cumulative = probabilities.cumsum(-1)
        cumulative[-1] = 1
        draw = torch.tensor(self.random.random(), device=logits.device, dtype=torch.float32)
        selected = torch.searchsorted(cumulative, draw, right=True).clamp_max(values.numel()-1)
        return selected if indices is None else indices[selected]


def sample_rows(logits, samplers, greedy=None):
    """Retain the existing batched argmax/readback for all-greedy traffic."""
    if all(not sampler.options.temperature and sampler.constraint is None for sampler in samplers):
        return (greedy if greedy is not None else logits.argmax(-1)).tolist()
    import torch
    return torch.stack([sampler.select(logits[i]) for i, sampler in enumerate(samplers)]).tolist()
