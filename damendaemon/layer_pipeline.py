"""Shared ownership for explicitly typed local CPU layer pipelines."""
from dataclasses import dataclass, field
import threading
import torch


@dataclass
class LayerPipelineState:
    owner: object
    stages: list
    capacity: int
    max_cache_bytes: int
    batch: int
    position: int = 0
    poisoned: bool = False
    closed: bool = False
    lock: object = field(default_factory=threading.RLock, repr=False)

    def _poison(self):
        self.poisoned = True
        for state in self.stages: state._poison()

    def _usable(self):
        if self.closed or self.poisoned or self.owner is None:
            raise RuntimeError('CPU layer pipeline cache is closed or poisoned')
        try:
            if len(self.stages) != len(self.owner.stages): raise RuntimeError('CPU layer pipeline lost a stage')
            for state, stage in zip(self.stages, self.owner.stages):
                state._usable()
                if (state.owner is not stage or state.position != self.position or state.capacity != self.capacity
                        or self.owner._stage_batch(state) != self.batch):
                    raise RuntimeError('CPU layer pipeline stage ownership/cursors disagree')
        except BaseException:
            self._poison(); raise

    def reorder(self, indices):
        with self.lock:
            self._usable()
            if (not isinstance(indices, torch.Tensor) or indices.ndim != 1 or not indices.numel()
                    or indices.numel() > 128 or indices.dtype != torch.int64 or indices.device.type != 'cpu'
                    or bool(((indices < 0) | (indices >= self.batch)).any())
                    or indices.numel() * self.owner.cache_bytes_per_row(self.capacity) > self.max_cache_bytes):
                raise ValueError('invalid CPU layer pipeline row order or aggregate cache budget')
            try:
                for state in self.stages: state.reorder(indices)
            except BaseException:
                self._poison(); raise
            self.batch = indices.numel()

    def close(self):
        with self.lock:
            self.closed = True
            pending = []; failure = None
            for state in self.stages:
                try: state.close()
                except BaseException as error:
                    pending.append(state)
                    if failure is None: failure = error
            self.stages = pending
            if not pending: self.owner = None
            if failure is not None: raise failure


class CpuLayerPipeline:
    """Prepared synchronous local pipeline, not a network/serving model.

    A partial execution or reorder poisons every stage in the epoch. Caller
    owns each returned state and must close it; failed prefill/allocation cleanup
    is retained here and retried before the next allocation.
    """
    stage_type = None

    def __init__(self, stages):
        if (self.stage_type is None or not isinstance(stages, (tuple, list)) or not 2 <= len(stages) <= 8
                or any(not isinstance(stage, self.stage_type) for stage in stages)
                or len({id(stage) for stage in stages}) != len(stages)):
            raise ValueError('CPU layer pipeline needs 2..8 distinct prepared stages')
        blocks = [block for stage in stages for block in stage.blocks]
        if (len({id(block) for block in blocks}) != len(blocks)
                or stages[0].start != 0 or stages[-1].end != stages[0].total
                or any(left.end != right.start for left, right in zip(stages, stages[1:]))
                or any(stage.total != stages[0].total or stage.dim != stages[0].dim for stage in stages)
                or stages[0].embedding.shape != self._head_shape(stages[-1])):
            raise ValueError('CPU layer pipeline needs complete contiguous compatible stage coverage')
        configs = [getattr(stage, 'config', None) for stage in stages]
        if any(config != configs[0] for config in configs): raise ValueError('CPU layer pipeline checkpoint configurations disagree')
        self.stages = tuple(stages)
        self.failed_cleanups = []
        self.lock = threading.RLock()

    def _head_shape(self,stage):return stage.head.shape

    @property
    def cleanup_pending(self):
        return bool(self.failed_cleanups or any(stage.failed_cleanups for stage in self.stages))

    @property
    def embedding(self):
        return self.stages[0].embedding

    @property
    def config(self):
        # Prepared stages without checkpoint metadata use the existing bounded
        # reference defaults; loaded stages were checked for identical config.
        import copy
        return copy.deepcopy(getattr(self.stages[0], 'config', {}))

    @property
    def boundary_width(self):
        return self.stages[0].dim

    def _stage_batch(self,state):
        raise NotImplementedError

    def cache_bytes_per_row(self,capacity):
        raise NotImplementedError

    def cleanup(self):
        with self.lock:
            pending = []; failures = []
            for state in self.failed_cleanups:
                try: state.close()
                except BaseException as error: pending.append(state); failures.append(error)
            self.failed_cleanups = pending
            for stage in self.stages:
                try: stage.cleanup()
                except BaseException as error: failures.append(error)
            if failures: raise failures[0]

    def _release_failed(self, state, error):
        with self.lock:
            self.failed_cleanups.append(state)
            try: self.cleanup()
            except BaseException:
                error.add_note('CPU layer pipeline cleanup incomplete; owners retained for retry')

    def new_state(self, batch, capacity, *, max_cache_bytes=256*1024**2):
        if (type(batch) is not int or not 1 <= batch <= 128
                or type(capacity) is not int or not 1 <= capacity <= 2**20
                or type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 2**31
                or batch * self.cache_bytes_per_row(capacity) > max_cache_bytes):
            raise ValueError('invalid CPU layer pipeline cache controls or aggregate budget exceeded')
        with self.lock:
            self.cleanup()
            state = LayerPipelineState(self, [], capacity, max_cache_bytes, batch)
            try:
                for stage in self.stages:
                    state.stages.append(stage.new_state(batch, capacity, max_cache_bytes=max_cache_bytes))
            except BaseException as error:
                self._release_failed(state, error); raise
            return state

    def _tokens(self, tokens, all_logits):
        if type(all_logits) is not bool: raise ValueError('CPU layer pipeline all_logits must be boolean')
        self.stages[0]._tokens(tokens, False)
        self._input_budget(tokens.shape,all_logits)

    def _input_budget(self,shape,all_logits):
        if (type(all_logits) is not bool or len(shape)!=2 or any(type(v) is not int or v<1 for v in shape)):
            raise ValueError('invalid CPU pipeline input geometry')
        batch, length = shape
        hidden_bytes = 4 * batch * length * self.boundary_width
        if any(hidden_bytes > stage.max_output_bytes for stage in self.stages):
            raise ValueError('CPU layer pipeline hidden boundary exceeds output budget')
        final_bytes = 4 * batch * (length if all_logits else 1) * self._head_shape(self.stages[-1])[0]
        if final_bytes > self.stages[-1].max_output_bytes:
            raise ValueError('CPU layer pipeline logits exceed output budget')

    @torch.no_grad()
    def forward(self, tokens, state, *, position, all_logits=False):
        self._tokens(tokens, all_logits)
        return self._forward(tokens,state,position=position,all_logits=all_logits)

    @torch.no_grad()
    def _forward(self,tokens,state,*,position,all_logits,first_forward=None):
        if not isinstance(state, LayerPipelineState) or state.owner is not self:
            raise ValueError('CPU layer pipeline cache belongs to another owner')
        with state.lock:
            state._usable()
            if (type(position) is not int or position != state.position
                    or tokens.shape[0] != state.batch or position + tokens.shape[1] > state.capacity):
                raise ValueError('CPU layer pipeline input/cursor/capacity mismatch')
            hidden = tokens
            try:
                for index,(stage,local) in enumerate(zip(self.stages,state.stages)):
                    forward=first_forward if index==0 and first_forward is not None else stage.forward
                    hidden=forward(hidden,local,position=position,all_logits=all_logits and stage.last)
            except BaseException:
                state._poison(); raise
            state.position = position + tokens.shape[1]
            return hidden

    def prefill(self, tokens, *, capacity, max_cache_bytes=256*1024**2, all_logits=False):
        self._tokens(tokens, all_logits)
        if type(capacity) is not int or tokens.shape[1] > capacity:
            raise ValueError('CPU layer pipeline prompt exceeds capacity')
        state = self.new_state(tokens.shape[0], capacity, max_cache_bytes=max_cache_bytes)
        try: return self.forward(tokens, state, position=0, all_logits=all_logits), state
        except BaseException as error:
            self._release_failed(state, error); raise
