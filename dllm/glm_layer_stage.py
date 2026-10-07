"""CPU reference GLM layer ranges with explicit sparse-indexer boundaries."""
import threading
import torch
import torch.nn.functional as F
from .glm_attention import _float, _budgets
from .glm_block import GlmBlock
from .glm_model import GlmModel, GlmModelState


def validate_range(start, end, total, indexer_types):
    if (any(type(value) is not int for value in (start, end, total))
            or not 0 <= start < end <= total <= 128
            or len(indexer_types) != total or indexer_types[start] != 'full'
            or (end < total and indexer_types[end] != 'full')):
        raise ValueError('GLM stage range must preserve full/shared indexer groups')


class GlmLayerStage:
    """One prepared CPU range; callers own the surrounding pipeline/transport.

    Reuses model state budgets, reorder, poisoning and retryable cleanup. Inputs
    are token IDs at the first stage and finite FP32 hidden values elsewhere.
    Every non-final stage returns the complete hidden chunk; only the final
    stage can select last-token versus all-token logits. No network is opened.
    """
    # Share the cache lifecycle without masquerading as a complete model in
    # generation/serving admission, which uses isinstance(GlmModel).
    cleanup = GlmModel.cleanup
    new_state = GlmModel.new_state
    _release_failed = GlmModel._release_failed
    _logits = GlmModel._logits
    prefill = GlmModel.prefill

    def __init__(self, blocks, *, start, total, embedding=None, norm=None, head=None,
                 epsilon=1e-5, max_output_bytes=64*1024**2):
        if (not isinstance(blocks, (tuple, list)) or not blocks
                or any(not isinstance(block, GlmBlock) for block in blocks)
                or type(start) is not int or type(total) is not int
                or not 0 <= start < start + len(blocks) <= total <= 128
                or blocks[0].attention.indexer is None
                or type(epsilon) not in (int, float) or not 0 < epsilon <= 1):
            raise ValueError('invalid prepared GLM layer stage')
        self.start, self.end, self.total = start, start + len(blocks), total
        self.first, self.last = start == 0, self.end == total
        self.dim = blocks[0].dim
        top = None
        for block in blocks:
            if block.dim != self.dim: raise ValueError('GLM stage block dimensions disagree')
            if block.attention.indexer is not None: top = block.attention.index_top_k
            elif block.attention.index_top_k != top: raise ValueError('GLM shared indexer top-k disagrees')
        if self.first:
            if not _float(embedding, 2) or embedding.shape[1] != self.dim:
                raise ValueError('first GLM stage requires its embedding')
        elif embedding is not None: raise ValueError('non-first GLM stage cannot own embedding')
        if self.last:
            if (not _float(norm, 1) or norm.shape != (self.dim,) or not _float(head, 2)
                    or head.shape[1] != self.dim or (self.first and head.shape != embedding.shape)):
                raise ValueError('final GLM stage requires norm and vocabulary head')
        elif norm is not None or head is not None: raise ValueError('non-final GLM stage cannot own norm/head')
        if any(not torch.isfinite(value).all() for value in (embedding, norm, head) if value is not None):
            raise ValueError('GLM stage boundary weights must be finite')
        _budgets(0, 0, max_output_bytes, 1)
        self.embedding, self.norm, self.head = embedding, norm, head
        self.blocks = tuple(blocks)
        self.epsilon, self.max_output_bytes = epsilon, max_output_bytes
        self.failed_cleanups = []
        self.lock = threading.RLock()

    def _tokens(self, inputs, all_logits):
        if type(all_logits) is not bool or (all_logits and not self.last):
            raise ValueError('only the final GLM stage produces all-token logits')
        if self.first:
            if (not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or min(inputs.shape) < 1
                    or inputs.dtype != torch.int64 or inputs.device.type != 'cpu'
                    or bool(((inputs < 0) | (inputs >= self.embedding.shape[0])).any())):
                raise ValueError('first GLM stage requires CPU vocabulary IDs')
        elif (not _float(inputs, 3) or min(inputs.shape) < 1 or inputs.shape[2] != self.dim
              or not torch.isfinite(inputs).all()):
            raise ValueError('GLM stage requires finite CPU FP32 hidden chunks')
        batch, length = inputs.shape[:2]
        required = 4 * batch * length * self.dim
        if self.last:
            required = max(required, 4 * batch * (length if all_logits else 1) * self.head.shape[0])
        if required > self.max_output_bytes: raise ValueError('GLM stage boundary exceeds output budget')

    def forward(self, inputs, state, *, position, all_logits=False):
        return self._forward(inputs, state, position=position, all_logits=all_logits, activations=False)

    def forward_activations(self, hidden, state, *, position):
        """Execute owned layers only, with no embedding or final projection."""
        return self._forward(hidden, state, position=position, all_logits=False, activations=True)

    @torch.no_grad()
    def _forward(self, inputs, state, *, position, all_logits, activations):
        if activations:
            if (not _float(inputs, 3) or inputs.shape[2] != self.dim
                    or inputs.numel()*4 > self.max_output_bytes or not torch.isfinite(inputs).all()):
                raise ValueError('GLM activation boundary requires bounded finite CPU FP32 hidden chunks')
        else: self._tokens(inputs, all_logits)
        if not isinstance(state, GlmModelState): raise ValueError('invalid GLM stage cache')
        with state.lock:
            state._usable()
            if (state.owner is not self or type(position) is not int or position != state.position
                    or inputs.shape[0] != state.layers[0].attention.latents.shape[0]
                    or position + inputs.shape[1] > state.capacity):
                raise ValueError('GLM stage cache owner/position/shape mismatch')
            hidden = F.embedding(inputs, self.embedding) if self.first and not activations else inputs
            selected = None
            try:
                for block, layer in zip(self.blocks, state.layers):
                    hidden, selected = block.forward(hidden, layer, position=position,
                        previous_indices=selected if block.attention.indexer is None else None)
                result = self._logits(hidden, all_logits) if self.last and not activations else hidden
            except BaseException:
                if any(layer.poisoned or layer.attention.poisoned or layer.position != position
                       or layer.attention.position != position for layer in state.layers): state._poison()
                raise
            state.position = position + inputs.shape[1]
            return result
