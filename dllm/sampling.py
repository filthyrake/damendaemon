"""Request-local random streams and device-side probability filtering."""
from collections import Counter
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0
    top_p: float = 1
    top_k: int = 0
    seed: int | None = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    repetition_penalty: float = 1
    logit_bias: dict | None = None

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

        for name in ('presence_penalty', 'frequency_penalty', 'repetition_penalty'):
            value = getattr(self, name)
            if type(value) not in (float,int) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if (name == 'repetition_penalty' and value <= 0) or (name != 'repetition_penalty' and not -2 <= value <= 2):
                raise ValueError('invalid token penalty')
        bias = self.logit_bias
        if bias is not None:
            if not isinstance(bias,dict) or len(bias) > 1024:
                raise ValueError('logit_bias must contain at most 1024 token entries')
            normalized = {}
            for key,value in bias.items():
                if isinstance(key,str) and key.isascii() and key.isdecimal() and str(int(key)) == key:
                    key = int(key)
                if (type(key) is not int or key < 0 or key in normalized
                        or type(value) not in (float,int) or not math.isfinite(value) or not -100 <= value <= 100):
                    raise ValueError('invalid logit bias token or value')
                normalized[key] = value
            object.__setattr__(self,'logit_bias',normalized)


class Sampler:
    def __init__(self, options):
        self.options = options
        self.constraint = None
        self.logprobs_top = None
        self.last_logprobs = None
        self.token_decoder = None
        self.prompt_tokens = set()
        self.counts = Counter()
        self.has_penalties = bool(options.presence_penalty or options.frequency_penalty or options.repetition_penalty != 1)
        self.has_adjustments = self.has_penalties or bool(options.logit_bias)
        # Never mutate the global RNG, or consume another request's stream.
        self.random = random.Random(options.seed) if options.temperature else None

    def prepare(self, prompt, vocab_size):
        if any(token >= vocab_size for token in (self.options.logit_bias or {})):
            raise ValueError('logit_bias token exceeds model vocabulary')
        if self.options.repetition_penalty != 1:
            self.prompt_tokens = set(prompt)
        self.counts.clear()

    def observe(self, token):
        if self.has_penalties:
            self.counts[token] += 1

    def adjust(self, logits):
        """Copy only on opt-in; graph output storage remains untouched."""
        if not self.has_adjustments:
            return logits
        import torch
        values = logits.float().clone()
        options = self.options
        if options.repetition_penalty != 1:
            seen = sorted(self.prompt_tokens | self.counts.keys())
            if seen:
                ids = torch.tensor(seen,device=values.device,dtype=torch.long)
                selected = values[ids]
                values[ids] = torch.where(selected > 0,selected/options.repetition_penalty,
                                          selected*options.repetition_penalty)
        if self.counts and (options.frequency_penalty or options.presence_penalty):
            ids = torch.tensor(list(self.counts),device=values.device,dtype=torch.long)
            counts = torch.tensor(list(self.counts.values()),device=values.device,dtype=torch.float32)
            values[ids] -= counts*options.frequency_penalty + options.presence_penalty
        if options.logit_bias:
            ids = torch.tensor(list(options.logit_bias),device=values.device,dtype=torch.long)
            bias = torch.tensor(list(options.logit_bias.values()),device=values.device,dtype=torch.float32)
            values[ids] += bias
        return values

    def select(self, logits):
        selected = self._select(logits)
        self.capture(logits,selected)
        return selected

    def capture(self, logits, selected):
        if self.logprobs_top is not None:
            from .logprobs import scores
            self.last_logprobs=scores(logits,selected,self.logprobs_top)

    def _select(self, logits):
        import torch
        options = self.options
        logits = self.adjust(logits)
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
    if all(not sampler.options.temperature and sampler.constraint is None and not sampler.has_adjustments for sampler in samplers):
        selected=greedy if greedy is not None else logits.argmax(-1)
    else:
        import torch
        selected=torch.stack([sampler._select(logits[i]) for i,sampler in enumerate(samplers)])
    scored=[i for i,sampler in enumerate(samplers) if sampler.logprobs_top is not None]
    if scored:
        from .logprobs import score_rows
        all_rows=len(scored)==len(samplers)
        packets=score_rows(logits if all_rows else logits[scored],selected if all_rows else selected[scored],
                           [samplers[i].logprobs_top for i in scored])
        for index,packet in zip(scored,packets):samplers[index].last_logprobs=packet
        if all_rows:return [packet['token_id'] for packet in packets]
    return selected.tolist()
