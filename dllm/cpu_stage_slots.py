"""Shared bounded CPU stage-slot ownership with explicit family hooks."""
import threading
import torch
from .glm_attention import _float


class CpuStageSlots:
    """Hidden-activation prerequisite for host commands; no transport admission.

    Reserve capacity for every slot before allocating any cache. Different slots
    may have different cursors. A partial compute/release failure discards the
    whole owner; retain it and retry close if release cannot finish.
    """
    stage_type=None

    def __init__(self, stage, context, max_batch=4, *, max_cache_bytes=256*1024**2):
        if (self.stage_type is None or not isinstance(stage, self.stage_type) or type(context) is not int
                or not 1 <= context <= min(2**20, self._configuration(stage).get('max_position_embeddings', 2**20))
                or type(max_batch) is not int or not 1 <= max_batch <= 4
                or type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 2**31):
            raise ValueError('invalid CPU stage slot controls')
        reserved = max_batch * self._cache_per_slot(stage,context)
        if reserved > max_cache_bytes: raise ValueError('CPU stage aggregate slot cache budget exceeded')
        self.stage, self.context, self.max_batch = stage, context, max_batch
        self.max_cache_bytes, self.reserved_cache_bytes = max_cache_bytes, reserved
        self.states = [None] * max_batch
        self.lengths = [0] * max_batch
        self.closed = self.failed = False
        self.lock = threading.RLock()

    actual_device = torch.device('cpu')
    compute_dtype = torch.float32

    @property
    def c(self):
        import copy
        return copy.deepcopy(getattr(self.stage, 'config', {}))

    @property
    def layer_start(self): return self.stage.start

    @property
    def layer_end(self): return self.stage.end

    @property
    def length(self): return self.lengths[0]

    @length.setter
    def length(self, value):
        if value != self.lengths[0]: raise ValueError('CPU slot zero cursor disagrees')

    def _usable(self):
        if self.closed or self.failed: raise RuntimeError('CPU stage slot owner is closed or failed')
        for index, state in enumerate(self.states):
            if state is None:
                if self.lengths[index]: raise RuntimeError('CPU stage slot ownership disagrees')
            elif state.owner is not self.stage or state.closed or state.poisoned or state.position != self.lengths[index]:
                raise RuntimeError('CPU stage slot ownership/cursor disagrees')

    def _ready(self):
        try: self._usable()
        except BaseException as error:
            if not self.closed: self._fail(error)
            raise

    def _fail(self, error):
        self.failed = True
        try: self.close()
        except BaseException: error.add_note('CPU stage slot cleanup incomplete; retain owner and retry close')

    @torch.no_grad()
    def embedding(self, tokens):
        with self.lock:
            self._ready()
            if not self.stage.first: raise ValueError('only the first CPU stage embeds tokens')
            self.stage._tokens(tokens, False)
            return self._embed(tokens)

    @torch.no_grad()
    def forward_layers(self, hidden, positions, slots, prefill, prefill_start=0):
        with self.lock:
            self._ready()
            if (not _float(hidden, 2+len(self._hidden_shape)) or tuple(hidden.shape[2:]) != self._hidden_shape
                    or hidden.shape[0] > self.max_batch or hidden.numel()*4 > self.stage.max_output_bytes
                    or not torch.isfinite(hidden).all()
                    or not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64
                    or positions.device.type != 'cpu' or positions.shape != hidden.shape[:2]
                    or not isinstance(slots, torch.Tensor) or slots.dtype != torch.int64
                    or slots.device.type != 'cpu' or slots.shape != (hidden.shape[0],)
                    or type(prefill) is not bool):
                raise ValueError('invalid CPU slot hidden/position/slot geometry')
            ids = slots.tolist(); starts = positions[:, 0].tolist(); length = hidden.shape[1]
            if (len(set(ids)) != len(ids) or any(not 0 <= slot < self.max_batch for slot in ids)
                    or (not prefill and length != 1)
                    or (prefill and (type(prefill_start) is not int or any(start != prefill_start for start in starts)))):
                raise ValueError('invalid CPU slot selection or prefill extent')
            for row, slot, start in zip(positions, ids, starts):
                if (start != self.lengths[slot] or start + length > self.context
                        or (not prefill and (self.states[slot] is None or start == 0))
                        or not torch.equal(row, torch.arange(start, start+length, dtype=torch.int64))):
                    raise ValueError('CPU slot cursor/positions/context disagree')
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
            if (not self.stage.last or not _float(hidden, 1+len(self._hidden_shape)) or tuple(hidden.shape[1:]) != self._hidden_shape
                    or hidden.shape[0] > self.max_batch or not torch.isfinite(hidden).all()
                    or hidden.shape[0]*self.stage.head.shape[0]*4 > self.stage.max_output_bytes):
                raise ValueError('invalid CPU slot final hidden/logit extent')
            try: return self.stage._logits(hidden[:, None], False)
            except BaseException as error:
                self._fail(error); raise

    def release(self, slots):
        with self.lock:
            self._ready()
            if (not isinstance(slots, (list, tuple)) or not slots
                    or any(type(slot) is not int or not 0 <= slot < self.max_batch or self.states[slot] is None for slot in slots)
                    or len(set(slots)) != len(slots)):
                raise ValueError('invalid occupied CPU slot release')
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
