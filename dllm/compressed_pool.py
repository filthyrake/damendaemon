"""Functional incremental pooling of projected DeepSeek compressed-KV channels.

Independent equations from DeepSeek-V4-Flash inference/model.py (MIT), revision
60d8d70770c6776ff598c94bb586a859a38244f1. Projection, normalization, RoPE and QAT
are separate. Values/gates/positional offsets are finite FP32 caller-owned inputs.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoolState:
    values: torch.Tensor
    scores: torch.Tensor
    position: int
    ratio: int

    def reorder(self, indices):
        """Copy selected batch slots; all slots in this state share a position."""
        if (not isinstance(indices, torch.Tensor) or indices.ndim != 1
                or indices.dtype != torch.int64 or indices.device != self.values.device):
            raise ValueError('pool reorder requires same-device int64 slot indices')
        return PoolState(self.values.index_select(0, indices), self.scores.index_select(0, indices),
                         self.position, self.ratio)


@torch.no_grad()
def pool_projected(values, gates, offsets, *, state=None):
    """Return (completed FP32 vectors, replacement state, block-start positions).

    values/gates [batch, tokens, channels], offsets [ratio, channels]. Ratio four
    uses overlapping blocks and channels=2*output_width; other ratios use ordinary
    disjoint pooling. In the overlapping case, the previous block contributes its
    first channel half and the current block its second half. The first previous
    block is absent, not zero-valued attention mass.

    This function never mutates inputs or the supplied state. The caller publishes
    the replacement only after downstream normalization/cache writes succeed.
    State holds at most two blocks; output size is proportional to completed
    blocks. Concurrent use requires the caller to serialize position commits.
    """
    if (not isinstance(values, torch.Tensor) or values.ndim != 3 or values.dtype != torch.float32
            or not isinstance(gates, torch.Tensor) or gates.shape != values.shape
            or gates.dtype != torch.float32 or gates.device != values.device
            or not isinstance(offsets, torch.Tensor) or offsets.ndim != 2 or offsets.dtype != torch.float32
            or offsets.device != values.device or offsets.shape[1] != values.shape[2]
            or not 1 <= offsets.shape[0] <= 128 or values.shape[2] < 1):
        raise ValueError('pooling requires matching FP32 values/gates and 1..128 positional offset rows')
    ratio, channels = offsets.shape
    overlap = ratio == 4
    if overlap and channels % 2:
        raise ValueError('overlapping compression requires two equal channel halves')
    width = channels//2 if overlap else channels
    batch, tokens = values.shape[:2]
    shape = (batch, ratio*(2 if overlap else 1), channels)
    if state is not None and (not isinstance(state, PoolState) or type(state.position) is not int
            or state.position < 0 or type(state.ratio) is not int or state.ratio != ratio
            or any(not isinstance(t, torch.Tensor) or t.shape != shape or t.dtype != torch.float32
                   or t.device != values.device for t in (state.values, state.scores))):
        raise ValueError('pool state geometry, position or device does not match input')
    position = state.position if state is not None else 0
    buffer = state.values.clone() if state is not None else values.new_zeros(shape)
    scores = state.scores.clone() if state is not None else values.new_full(shape, -float('inf'))
    count = (position+tokens)//ratio-position//ratio
    output = values.new_empty(batch, count, width)
    starts = torch.arange(position//ratio, (position+tokens)//ratio, device=values.device)*ratio
    current = ratio if overlap else 0
    consumed = completed = 0
    while consumed < tokens:
        partial = (position+consumed) % ratio
        take = min(tokens-consumed, ratio-partial)
        target = slice(current+partial, current+partial+take)
        buffer[:, target] = values[:, consumed:consumed+take]
        scores[:, target] = gates[:, consumed:consumed+take]+offsets[partial:partial+take]
        consumed += take
        if (position+consumed) % ratio == 0:
            if overlap:
                pooled_values = torch.cat((buffer[:, :ratio, :width], buffer[:, ratio:, width:]), dim=1)
                pooled_scores = torch.cat((scores[:, :ratio, :width], scores[:, ratio:, width:]), dim=1)
            else:
                pooled_values, pooled_scores = buffer, scores
            output[:, completed] = (pooled_values*pooled_scores.softmax(dim=1)).sum(dim=1)
            completed += 1
            if overlap:
                buffer[:, :ratio].copy_(buffer[:, ratio:])
                scores[:, :ratio].copy_(scores[:, ratio:])
            buffer[:, current:].zero_()
            scores[:, current:].fill_(-float('inf'))
    return output, PoolState(buffer, scores, position+tokens, ratio), starts
