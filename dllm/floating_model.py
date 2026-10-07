"""Prepared-floating DeepSeek reference stack; no packed loader or serving admission."""
from dataclasses import dataclass, field
import threading

import torch
from torch.nn import functional as F

from .floating_attention import _norm
from .floating_block import FloatingBlock
from .hyper_connections import reduce_head


@dataclass
class ModelState:
    owner: object
    layers: list
    position: int
    capacity: int
    max_cache_bytes: int
    poisoned: bool = False
    closed: bool = False
    lock: object = field(default_factory=threading.RLock, repr=False)

    def _poison(self):
        self.poisoned = True
        for layer in self.layers:
            layer.attention.poisoned = True

    def _usable(self):
        if self.closed or self.poisoned:
            raise RuntimeError('model cache is closed or poisoned')
        if (self.owner is None or len(self.layers) != len(self.owner.blocks)
                or any(s.owner is not block or s.attention.closed or s.attention.poisoned
                       or s.attention.position != self.position
                       for block, s in zip(self.owner.blocks, self.layers))):
            self._poison()
            raise RuntimeError('model layer caches disagree or are unusable')

    def reorder(self, indices):
        with self.lock:
            self._usable()
            first = self.layers[0].attention
            if (not isinstance(indices,torch.Tensor) or indices.ndim != 1 or indices.numel() < 1
                    or indices.dtype != torch.int64 or indices.device != first.cache.device
                    or indices.numel()*sum(s.attention.bytes_per_slot for s in self.layers) > self.max_cache_bytes):
                raise ValueError('invalid model slot order or aggregate cache budget exceeded')
            changed = False
            try:
                for layer in self.layers:
                    layer.reorder(indices)
                    changed = True
            except BaseException:
                if changed:
                    self._poison()
                raise

    def close(self):
        """Best-effort release with retryable ownership of any failed cleanup."""
        with self.lock:
            self.closed = True
            pending, error = [], None
            for layer in self.layers:
                try:
                    layer.close()
                except BaseException as failure:
                    pending.append(layer)
                    if error is None:
                        error = failure
            self.layers = pending
            if not pending:
                self.owner = None
            if error is not None:
                raise error


class FloatingModel:
    """Own prepared floating blocks, embedding and final head for CPU reference work.

    No real-checkpoint equivalence or GPU qualification follows from tiny fixture
    execution. Embedding/projected block parameters retain their prepared dtype;
    the final vocabulary projection and head mixing/norm parameters are explicit
    FP32 tensors. Callers must not mutate prepared parameter contents.
    """
    def __init__(self, embedding, blocks, head, *, epsilon=1e-6, connection_epsilon=1e-6):
        if (not isinstance(blocks,(list,tuple)) or not blocks or any(not isinstance(b,FloatingBlock) for b in blocks)
                or not isinstance(embedding,torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] < 1):
            raise ValueError('model needs prepared blocks and a vocabulary embedding matrix')
        first = blocks[0]
        if embedding.shape[1] != first.dim or embedding.dtype != first.dtype or embedding.device != first.device:
            raise ValueError('embedding geometry/dtype/device must match prepared blocks')
        for block in blocks:
            if (block.dim != first.dim or block.hc_mult != first.hc_mult or block.dtype != first.dtype or block.device != first.device
                    or (block.table is not None and block.table.shape[0] != embedding.shape[0])):
                raise ValueError('block stream geometry, placement or hash vocabulary differs')
        specs = {'head.weight':tuple(embedding.shape), 'norm.weight':(first.dim,),
                 'hc_head_fn':(first.hc_mult,first.hc_mult*first.dim), 'hc_head_scale':(1,), 'hc_head_base':(first.hc_mult,)}
        if not isinstance(head,dict) or set(head) != set(specs):
            raise ValueError('invalid model head parameter names')
        for name,shape in specs.items():
            tensor = head[name]
            if not isinstance(tensor,torch.Tensor) or tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != embedding.device:
                raise ValueError(f'invalid FP32 model head parameter: {name}')
        if any(type(v) not in (int,float) or not 0 < v <= torch.finfo(torch.float32).max for v in (epsilon,connection_epsilon)):
            raise ValueError('model normalization epsilons must be finite and positive')
        self.embedding, self.blocks, self.head = embedding, tuple(blocks), dict(head)
        self.epsilon, self.connection_epsilon = epsilon, connection_epsilon
        self.failed_prefills = []
        self.build_lock = threading.RLock()

    def _embed(self, token_ids):
        if (not isinstance(token_ids,torch.Tensor) or token_ids.ndim != 2 or min(token_ids.shape) < 1
                or token_ids.dtype not in (torch.int32,torch.int64) or token_ids.device != self.embedding.device):
            raise ValueError('model token IDs must be nonempty same-device integer [batch,sequence]')
        values = self.embedding.index_select(0,token_ids.reshape(-1).long()).reshape(*token_ids.shape,-1)
        return values.unsqueeze(2).expand(-1,-1,self.blocks[0].hc_mult,-1)

    def _logits(self, hidden):
        h = self.head
        reduced = reduce_head(hidden[:,-1:],h['hc_head_fn'],h['hc_head_scale'],h['hc_head_base'],
                              norm_epsilon=self.epsilon,connection_epsilon=self.connection_epsilon)
        normalized = _norm(reduced,h['norm.weight'],self.epsilon)
        return F.linear(normalized[:,0].float(),h['head.weight'])

    def cleanup_failed_prefills(self):
        with self.build_lock:
            while self.failed_prefills:
                self.failed_prefills[0].close()
                self.failed_prefills.pop(0)

    @torch.no_grad()
    def prefill(self, token_ids, *, max_tokens=None, max_cache_bytes=256*1024*1024, query_chunk=16):
        with self.build_lock:
            self.cleanup_failed_prefills()
            if type(max_cache_bytes) is not int or max_cache_bytes < 1:
                raise ValueError('aggregate model cache budget must be a positive integer')
            hidden = self._embed(token_ids)
            length = token_ids.shape[1]
            capacity = length if max_tokens is None else max_tokens
            if type(capacity) is not int or not length <= capacity < 2**31:
                raise ValueError('invalid model cache capacity')
            layers, remaining = [], max_cache_bytes
            try:
                for block in self.blocks:
                    hidden,state = block.prefill(hidden,token_ids,max_tokens=capacity,max_cache_bytes=remaining,query_chunk=query_chunk)
                    layers.append(state)
                    remaining -= state.attention.bytes_per_slot*token_ids.shape[0]
                    # The model enforces the aggregate budget on every reorder.
                    # Construction's remaining budget must not become a smaller
                    # permanent limit for later layers when slots are expanded.
                    state.attention.max_cache_bytes = max_cache_bytes
                logits = self._logits(hidden)
            except BaseException as failure:
                pending = ModelState(self,layers,length,capacity,max_cache_bytes)
                try:
                    pending.close()
                except BaseException:
                    self.failed_prefills.append(pending)
                    failure.add_note('failed prefill cleanup remains owned and retryable before another prefill')
                raise
            return logits,ModelState(self,layers,length,capacity,max_cache_bytes)

    @torch.no_grad()
    def decode(self, token_ids, state, *, position):
        if not isinstance(state,ModelState) or state.owner is not self:
            raise ValueError('model cache belongs to a different or closed owner')
        with state.lock:
            state._usable()
            if type(position) is not int or position != state.position or position >= state.capacity:
                raise ValueError('stale model position or capacity exhausted')
            hidden = self._embed(token_ids)
            if token_ids.shape != (state.layers[0].attention.cache.shape[0],1):
                raise ValueError('model decode requires exactly one token per prepared slot')
            try:
                for block,layer in zip(self.blocks,state.layers):
                    hidden = block.decode(hidden,layer,token_ids,position=position)
                logits = self._logits(hidden)
            except BaseException:
                if any(layer.attention.position != position or layer.attention.poisoned for layer in state.layers):
                    state._poison()
                raise
            state.position = position+1
            return logits
