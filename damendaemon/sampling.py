"""Request-local random streams and device-side probability filtering."""
from collections import Counter
from dataclasses import dataclass
import math
import random


def _finite_number(value):
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False  # JSON integers can exceed the host float range.


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
            if not _finite_number(value):
                raise ValueError(f'{name} must be a finite number')
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError('temperature must be nonnegative and top_p must be in (0, 1]')
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError('seed must be an integer in [0, 2**64)')

        for name in ('presence_penalty', 'frequency_penalty', 'repetition_penalty'):
            value = getattr(self, name)
            if not _finite_number(value):
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
                        or not _finite_number(value) or not -100 <= value <= 100):
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
        # Finite request parameters can overflow FP32 penalty arithmetic.
        # Check this opt-in path before argmax can silently hide inf/NaN.
        # Existing -inf masks remain excluded under finite penalties/bias.
        # Only preserve those entries; overflow from a finite source is invalid.
        preserved_mask=torch.isneginf(logits) & torch.isneginf(values)
        if not (torch.isfinite(values) | preserved_mask).all().item():
            # Distinguish corrupt model output from request-induced overflow.
            # This extra readback is only needed on the failure path.
            if (torch.isnan(logits) | torch.isposinf(logits)).any().item():
                raise RuntimeError('cannot sample non-finite model logits')
            raise ValueError('cannot sample non-finite adjusted logits')
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
            maximum, selected = logits.max(-1)
            # max propagates NaN and preserves +inf/all-masked -inf. A finite
            # maximum still permits ordinary -inf vocabulary/grammar masks.
            if not torch.isfinite(maximum).item():
                raise RuntimeError('cannot sample non-finite greedy logits')
            return selected
        values = logits.float()
        indices = None
        if options.top_k:
            values, indices = values.topk(min(options.top_k, values.numel()))
        if options.top_p < 1:
            values, order = values.sort(descending=True)
            indices = order if indices is None else indices[order]
        if options.temperature >= 1:
            # Scale before centering: extreme finite logits can overflow their
            # difference even when the temperature-scaled difference is small.
            # Reciprocal multiplication also avoids narrowing huge temperatures
            # to FP32 infinity. Preserve masks if the reciprocal rounds to zero.
            scaled=torch.where(torch.isneginf(values),values,values*(1/options.temperature))
        else:
            # Center first so tiny temperatures cannot overflow positive logits.
            scaled=(values-values.max())/max(options.temperature,torch.finfo(torch.float32).tiny)
        probabilities = torch.softmax(scaled, -1)
        if options.top_p < 1:
            # Keep the first token that crosses the cumulative probability bound.
            remove = probabilities.cumsum(-1) - probabilities >= options.top_p
            probabilities = probabilities.masked_fill(remove, 0)
            probabilities = probabilities / probabilities.sum()
        if not torch.isfinite(probabilities).all().item():
            raise RuntimeError('cannot sample non-finite model probabilities')
        cumulative = probabilities.cumsum(-1)
        # Renormalize the entire CDF: assigning only its last entry to one
        # would give a zero-probability tail any FP32 rounding shortfall.
        cumulative = cumulative / cumulative[-1]
        cumulative[-1] = 1
        # A Python draw below one can round to exactly one in FP32.
        draw = torch.tensor(self.random.random(), device=logits.device, dtype=torch.float32).clamp_max(
            1 - torch.finfo(torch.float32).eps / 2)
        selected = torch.searchsorted(cumulative, draw, right=True).clamp_max(values.numel()-1)
        return selected if indices is None else indices[selected]


def _token_batch(tokens, count, vocab_size):
    # Validate the existing host readback before scheduler zip loops can drop
    # unmatched requests. Do not introduce another device synchronization.
    if (not isinstance(tokens, list) or len(tokens) != count
            or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens)):
        raise RuntimeError('invalid selected token batch')
    return tokens


def greedy_tokens(logits, selected=None):
    """On-device greedy IDs; -1 marks NaN/+inf/all-masked rows for readback.

    No host synchronization: callers must reject the sentinel before emitting
    tokens. Existing negative-infinity masks are valid if a finite maximum exists.
    """
    import torch
    if selected is None:
        maximum, selected = logits.max(-1)
    else:
        maximum=logits.amax(-1)
    if selected.shape != logits.shape[:-1] or selected.dtype not in (
            torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
        raise RuntimeError('invalid selected token batch')
    return selected.to(torch.long).masked_fill(~torch.isfinite(maximum),-1)


def sample_rows(logits, samplers, greedy=None):
    """Validate greedy rows on-device before the existing token readback."""
    import torch
    if (len(logits.shape) != 2 or logits.shape[0] != len(samplers)
            or logits.shape[1] == 0):
        raise RuntimeError('invalid sampling logit batch')
    if all(not sampler.options.temperature and sampler.constraint is None and not sampler.has_adjustments for sampler in samplers):
        selected=greedy_tokens(logits,greedy)
    else:
        import torch
        selected=torch.stack([sampler._select(logits[i]) for i,sampler in enumerate(samplers)])
    scored=[i for i,sampler in enumerate(samplers) if sampler.logprobs_top is not None]
    if scored:
        from .logprobs import score_rows
        all_rows=len(scored)==len(samplers)
        packets=score_rows(logits if all_rows else logits[scored],selected if all_rows else selected[scored],
                           [samplers[i].logprobs_top for i in scored])
        if (not isinstance(packets, list)
                or any(not isinstance(packet, dict) or 'token_id' not in packet for packet in packets)):
            raise RuntimeError('invalid scored token batch')
        scored_tokens = _token_batch([packet['token_id'] for packet in packets], len(scored), logits.shape[-1])
        tokens = scored_tokens if all_rows else _token_batch(selected.tolist(), len(samplers), logits.shape[-1])
        if not all_rows and scored_tokens != [tokens[index] for index in scored]:
            raise RuntimeError('scored tokens disagree with selected token batch')
        for index,packet in zip(scored,packets):samplers[index].last_logprobs=packet
        return tokens
    return _token_batch(selected.tolist(), len(samplers), logits.shape[-1])
