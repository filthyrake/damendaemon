"""Prepared CPU FP32 Flash mixed stack with aggregate cache ownership."""
from dataclasses import dataclass,field
import threading
import torch
import torch.nn.functional as F
from .glm_attention import _float,_budgets
from .glm_flash_block import FlashBlock
from .glm_flash_linear import FlashLinearAttention
from .glm_flash_sparse import FlashSparseState


def _batch(state):
    return (state.latents if isinstance(state,FlashSparseState) else state.recurrent).shape[0]


@dataclass
class FlashModelState:
    owner: object
    layers: list
    capacity: int
    max_cache_bytes: int
    position: int = 0
    poisoned: bool = False
    closed: bool = False
    lock: object = field(default_factory=threading.RLock,repr=False)

    def _poison(self):
        self.poisoned=True
        for layer in self.layers:layer.poisoned=True

    def _usable(self):
        if self.closed or self.poisoned or self.owner is None:
            raise RuntimeError('Flash model cache is closed or poisoned')
        if (len(self.layers)!=len(self.owner.blocks)
                or any(layer.owner is not block or layer.poisoned or layer.attention.owner is not block.attention
                       or layer.attention.closed
                       or layer.position!=self.position or layer.attention.position!=self.position
                       or layer.attention.capacity!=self.capacity
                       or _batch(layer.attention)!=_batch(self.layers[0].attention)
                       for layer,block in zip(self.layers,self.owner.blocks))):
            self._poison()
            raise RuntimeError('Flash model layer cache ownership/cursors disagree')

    def reorder(self,indices):
        with self.lock:
            self._usable()
            batch=_batch(self.layers[0].attention)
            per_slot=self.owner._cache_per_slot(self.capacity)
            if (not isinstance(indices,torch.Tensor) or indices.ndim!=1 or not 1<=indices.numel()<=128
                    or indices.dtype!=torch.int64 or indices.device.type!='cpu'
                    or bool(((indices<0)|(indices>=batch)).any()) or indices.numel()*per_slot>self.max_cache_bytes):
                raise ValueError('invalid Flash model slot order or aggregate cache budget exceeded')
            changed=False
            try:
                for layer in self.layers:
                    layer.reorder(indices);changed=True
            except BaseException:
                if changed:self._poison()
                raise

    def close(self):
        with self.lock:
            self.closed=True
            pending=[];failure=None
            for layer in self.layers:
                try:layer.close()
                except BaseException as error:
                    pending.append(layer)
                    if failure is None:failure=error
            self.layers=pending
            if not pending:self.owner=None
            if failure is not None:raise failure


class FlashModel:
    """Caller-owned prepared FP32 parameters; no checkpoint or serving loader.

    Keep parameter contents immutable. Cache budget is aggregate retained tensor
    storage; temporary reorder copies and compute scratch have separate bounds.
    """
    def __init__(self,embedding,blocks,norm,head,*,epsilon=1e-5,max_output_bytes=64*1024**2):
        if (not _float(embedding,2) or not _float(norm,1) or not _float(head,2)
                or head.shape!=embedding.shape or norm.shape!=(embedding.shape[1],)
                or not isinstance(blocks,(tuple,list)) or not 1<=len(blocks)<=128
                or any(not isinstance(block,FlashBlock) or block.dim!=embedding.shape[1] for block in blocks)
                or type(epsilon) not in (int,float) or not 0<epsilon<=1):
            raise ValueError('invalid prepared Flash model geometry')
        if any(block.streams!=blocks[0].streams for block in blocks):
            raise ValueError('Flash blocks must agree on residual stream count')
        self.streams=blocks[0].streams
        if any(not torch.isfinite(value).all() for value in (embedding,norm,head)):
            raise ValueError('Flash embedding/head must be finite')
        _budgets(0,0,max_output_bytes,1)
        self.embedding,self.blocks,self.norm,self.head=embedding,tuple(blocks),norm,head
        self.epsilon,self.max_output_bytes=epsilon,max_output_bytes
        self.failed_cleanups=[]
        self.lock=threading.RLock()

    def cleanup(self):
        with self.lock:
            while self.failed_cleanups:
                self.failed_cleanups[0].close()
                self.failed_cleanups.pop(0)

    def _tokens(self,tokens,all_logits):
        if (not isinstance(tokens,torch.Tensor) or tokens.ndim!=2 or min(tokens.shape)<1
                or tokens.dtype!=torch.int64 or tokens.device.type!='cpu'
                or bool(((tokens<0)|(tokens>=self.embedding.shape[0])).any()) or type(all_logits) is not bool):
            raise ValueError('Flash tokens require nonempty CPU int64 vocabulary IDs')
        rows=tokens.numel() if all_logits else tokens.shape[0]
        if max(4*tokens.numel()*self.embedding.shape[1]*self.streams,4*rows*self.embedding.shape[0])>self.max_output_bytes:
            raise ValueError('Flash embedding/logit output exceeds configured tensor storage budget')

    def _cache_per_slot(self,capacity):
        return sum(block.attention.bytes_per_slot if isinstance(block.attention,FlashLinearAttention)
                   else capacity*block.attention.bytes_per_token for block in self.blocks)

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        if (type(batch) is not int or not 1<=batch<=128 or type(capacity) is not int or not 1<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31):
            raise ValueError('invalid Flash model cache bounds')
        if batch*self._cache_per_slot(capacity)>max_cache_bytes:
            raise ValueError('aggregate Flash cache reservation exceeds budget')
        with self.lock:
            self.cleanup()
            state=FlashModelState(self,[],capacity,max_cache_bytes)
            try:
                for block in self.blocks:state.layers.append(block.new_state(batch,capacity,max_cache_bytes=max_cache_bytes))
            except BaseException as error:
                self._release_failed(state,error)
                raise
            return state

    def _release_failed(self,state,error):
        try:state.close()
        except BaseException:
            with self.lock:self.failed_cleanups.append(state)
            error.add_note('Flash cache cleanup failed; retained for retry before the next allocation')

    def _logits(self,hidden,all_logits):
        value=hidden if all_logits else hidden[:,-1]
        value=value.mean(-2)
        variance=value.square().mean(-1,keepdim=True)
        if not torch.isfinite(value).all() or not torch.isfinite(variance).all():
            raise ValueError('nonfinite Flash final normalization')
        value=value*torch.rsqrt(variance+self.epsilon)*self.norm
        result=F.linear(value,self.head)
        if not torch.isfinite(result).all():raise ValueError('nonfinite Flash model logits')
        return result

    @torch.no_grad()
    def forward(self,tokens,state,*,position,all_logits=False):
        self._tokens(tokens,all_logits)
        if not isinstance(state,FlashModelState):raise ValueError('invalid Flash model cache')
        with state.lock:
            state._usable()
            if (state.owner is not self or type(position) is not int or position!=state.position
                    or tokens.shape[0]!=_batch(state.layers[0].attention)
                    or position+tokens.shape[1]>state.capacity):
                raise ValueError('Flash model cache owner/position/shape mismatch')
            hidden=F.embedding(tokens,self.embedding).unsqueeze(-2).expand(-1,-1,self.streams,-1)
            try:
                for block,layer in zip(self.blocks,state.layers):
                    hidden=block.forward(hidden,layer,position=position)
                logits=self._logits(hidden,all_logits)
            except BaseException:
                if any(layer.poisoned or layer.position!=position
                       or layer.attention.position!=position for layer in state.layers):state._poison()
                raise
            state.position=position+tokens.shape[1]
            return logits

    def prefill(self,tokens,*,capacity,max_cache_bytes=256*1024**2,all_logits=False):
        self._tokens(tokens,all_logits)
        if type(capacity) is not int or tokens.shape[1]>capacity:raise ValueError('Flash prefill exceeds capacity')
        state=self.new_state(tokens.shape[0],capacity,max_cache_bytes=max_cache_bytes)
        try:return self.forward(tokens,state,position=0,all_logits=all_logits),state
        except BaseException as error:
            self._release_failed(state,error)
            raise
