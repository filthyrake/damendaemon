"""Independent bounded CPU cache slots around one prepared GLM layer stage."""
import threading
import torch
import torch.nn.functional as F
from .glm_attention import _float
from .glm_layer_stage import GlmLayerStage


class GlmStageSlots:
    """Hidden-activation prerequisite for host commands; no transport admission.

    Reserve capacity for every slot before allocating any cache. Different slots
    may have different cursors. A partial compute/release failure discards the
    whole owner; retain it and retry close if release cannot finish.
    """
    def __init__(self, stage, context, max_batch=4, *, max_cache_bytes=256*1024**2):
        if (not isinstance(stage, GlmLayerStage) or type(context) is not int
                or not 1 <= context <= min(2**20, getattr(stage, 'config', {}).get('max_position_embeddings', 2**20))
                or type(max_batch) is not int or not 1 <= max_batch <= 4
                or type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 2**31):
            raise ValueError('invalid GLM stage slot controls')
        width = sum(block.attention.rank + block.attention.rope_dim
            + (block.attention.indexer.width if block.attention.indexer is not None else 0) for block in stage.blocks)
        reserved = 4 * context * max_batch * width
        if reserved > max_cache_bytes: raise ValueError('GLM stage aggregate slot cache budget exceeded')
        self.stage, self.context, self.max_batch = stage, context, max_batch
        self.max_cache_bytes, self.reserved_cache_bytes = max_cache_bytes, reserved
        self.states = [None] * max_batch
        self.lengths = [0] * max_batch
        self.closed = self.failed = False
        self.lock = threading.RLock()

    def _usable(self):
        if self.closed or self.failed: raise RuntimeError('GLM stage slot owner is closed or failed')
        for index, state in enumerate(self.states):
            if state is None:
                if self.lengths[index]: raise RuntimeError('GLM stage slot ownership disagrees')
            elif state.owner is not self.stage or state.closed or state.poisoned or state.position != self.lengths[index]:
                raise RuntimeError('GLM stage slot ownership/cursor disagrees')

    def _ready(self):
        try: self._usable()
        except BaseException as error:
            if not self.closed: self._fail(error)
            raise

    def _fail(self, error):
        self.failed = True
        try: self.close()
        except BaseException: error.add_note('GLM stage slot cleanup incomplete; retain owner and retry close')

    @torch.no_grad()
    def embedding(self, tokens):
        with self.lock:
            self._ready()
            if not self.stage.first: raise ValueError('only the first GLM stage embeds tokens')
            self.stage._tokens(tokens, False)
            return F.embedding(tokens, self.stage.embedding)

    @torch.no_grad()
    def forward_layers(self, hidden, positions, slots, prefill, prefill_start=0):
        with self.lock:
            self._ready()
            if (not _float(hidden, 3) or hidden.shape[2] != self.stage.dim
                    or hidden.shape[0] > self.max_batch or hidden.numel()*4 > self.stage.max_output_bytes
                    or not torch.isfinite(hidden).all()
                    or not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64
                    or positions.device.type != 'cpu' or positions.shape != hidden.shape[:2]
                    or not isinstance(slots, torch.Tensor) or slots.dtype != torch.int64
                    or slots.device.type != 'cpu' or slots.shape != (hidden.shape[0],)
                    or type(prefill) is not bool):
                raise ValueError('invalid GLM slot hidden/position/slot geometry')
            ids = slots.tolist(); starts = positions[:, 0].tolist(); length = hidden.shape[1]
            if (len(set(ids)) != len(ids) or any(not 0 <= slot < self.max_batch for slot in ids)
                    or (not prefill and length != 1)
                    or (prefill and (type(prefill_start) is not int or any(start != prefill_start for start in starts)))):
                raise ValueError('invalid GLM slot selection or prefill extent')
            for row, slot, start in zip(positions, ids, starts):
                if (start != self.lengths[slot] or start + length > self.context
                        or (not prefill and (self.states[slot] is None or start == 0))
                        or not torch.equal(row, torch.arange(start, start+length, dtype=torch.int64))):
                    raise ValueError('GLM slot cursor/positions/context disagree')
            outputs = []
            try:
                for slot in ids:
                    if self.states[slot] is None:
                        self.states[slot] = self.stage.new_state(1, self.context, max_cache_bytes=self.max_cache_bytes)
                for row, (slot, start) in enumerate(zip(ids, starts)):
                    outputs.append(self.stage.forward_activations(hidden[row:row+1], self.states[slot], position=start))
                result = torch.cat(outputs, dim=0)
            except BaseException as error:
                self._fail(error); raise
            for slot, start in zip(ids, starts): self.lengths[slot] = start + length
            return result

    @torch.no_grad()
    def output_logits(self, hidden):
        with self.lock:
            self._ready()
            if (not self.stage.last or not _float(hidden, 2) or hidden.shape[1] != self.stage.dim
                    or hidden.shape[0] > self.max_batch or not torch.isfinite(hidden).all()
                    or hidden.shape[0]*self.stage.head.shape[0]*4 > self.stage.max_output_bytes):
                raise ValueError('invalid GLM slot final hidden/logit extent')
            try: return self.stage._logits(hidden[:, None], False)
            except BaseException as error:
                self._fail(error); raise

    def release(self, slots):
        with self.lock:
            self._ready()
            if (not isinstance(slots, (list, tuple)) or not slots
                    or any(type(slot) is not int or not 0 <= slot < self.max_batch or self.states[slot] is None for slot in slots)
                    or len(set(slots)) != len(slots)):
                raise ValueError('invalid occupied GLM slot release')
            try:
                for slot in slots:
                    self.states[slot].close(); self.states[slot] = None; self.lengths[slot] = 0
            except BaseException as error:
                self._fail(error); raise

    def close(self):
        with self.lock:
            self.closed = True
            failures = []
            for slot, state in enumerate(self.states):
                if state is not None:
                    try: state.close(); self.states[slot] = None; self.lengths[slot] = 0
                    except BaseException as error: failures.append(error)
            try: self.stage.cleanup()
            except BaseException as error: failures.append(error)
            if failures: raise failures[0]
