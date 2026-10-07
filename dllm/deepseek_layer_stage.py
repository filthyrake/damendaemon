"""Explicit packed DeepSeek CPU ranges carrying tokens and residual streams."""
import threading
import torch
from torch.nn import functional as F
from .floating_block import FloatingBlock
from .floating_model import FloatingModel,ModelState
from .glm_attention import _float


def validate_range(start,end,total):
    if any(type(value) is not int for value in (start,end,total)) or not 0<=start<end<=total<=128:
        raise ValueError('DeepSeek stage requires a bounded contiguous nonempty layer range')


class DeepSeekStageState(ModelState):
    def _usable(self):
        super()._usable()
        batch=self.layers[0].attention.cache.shape[0]
        if any(local.attention.capacity!=self.capacity or local.attention.cache.shape[0]!=batch for local in self.layers):
            self._poison();raise RuntimeError('DeepSeek stage cache capacity or batch ownership differs')

    def reorder(self,indices):
        if (not isinstance(indices,torch.Tensor) or indices.dtype!=torch.int64 or indices.device.type!='cpu'
                or indices.ndim!=1 or not 1<=indices.numel()<=128):
            raise ValueError('DeepSeek stage reorder requires 1..128 CPU integer indices')
        with self.lock:
            self._usable();batch=self.layers[0].attention.cache.shape[0]
            if bool(((indices<0)|(indices>=batch)).any()):raise ValueError('DeepSeek stage reorder index out of bounds')
            return super().reorder(indices)


class DeepSeekLayerStage:
    """Separate from whole-model admission; callers own immutable prepared weights."""
    cleanup_failed_prefills=FloatingModel.cleanup_failed_prefills
    prefill_cleanup_pending=FloatingModel.prefill_cleanup_pending
    _logits=FloatingModel._logits

    def __init__(self,blocks,*,start,total,vocab_size,embedding=None,head=None,
                 epsilon=1e-6,connection_epsilon=1e-6,max_output_bytes=64*1024**2):
        if (not isinstance(blocks,(tuple,list)) or not blocks or any(not isinstance(b,FloatingBlock) for b in blocks)
                or type(vocab_size) is not int or not 1<=vocab_size<=2**21
                or type(max_output_bytes) is not int or not 1<=max_output_bytes<=2**31
                or any(type(v) not in (int,float) or not 0<v<=torch.finfo(torch.float32).max for v in (epsilon,connection_epsilon))):
            raise ValueError('invalid DeepSeek stage controls')
        validate_range(start,start+len(blocks),total)
        self.start,self.end,self.total=start,start+len(blocks),total
        self.first,self.last=start==0,self.end==total
        self.dim,self.streams=blocks[0].dim,blocks[0].hc_mult
        if any(b.device.type!='cpu' or b.dtype!=torch.float32 or b.dim!=self.dim or b.hc_mult!=self.streams
               or (b.table is not None and b.table.shape[0]!=vocab_size) for b in blocks):
            raise ValueError('DeepSeek stage block placement/stream/hash geometry disagrees')
        if self.first:
            if not _float(embedding,2) or embedding.shape!=(vocab_size,self.dim) or not torch.isfinite(embedding).all():
                raise ValueError('first DeepSeek stage needs a finite CPU FP32 embedding')
        elif embedding is not None:raise ValueError('only the first DeepSeek stage owns embeddings')
        if self.last:
            specs={'head.weight':(vocab_size,self.dim),'norm.weight':(self.dim,),
                   'hc_head_fn':(self.streams,self.streams*self.dim),'hc_head_scale':(1,),'hc_head_base':(self.streams,)}
            if not isinstance(head,dict) or set(head)!=set(specs):raise ValueError('final DeepSeek stage needs its complete head')
            for name,shape in specs.items():
                value=head[name]
                if not _float(value,len(shape)) or tuple(value.shape)!=shape or not torch.isfinite(value).all():
                    raise ValueError('invalid DeepSeek stage head geometry or values')
            head=dict(head)
        elif head is not None:raise ValueError('only the final DeepSeek stage owns the head')
        self.blocks,self.embedding,self.head=tuple(blocks),embedding,head
        self.vocab_size,self.max_output_bytes=vocab_size,max_output_bytes
        self.epsilon,self.connection_epsilon=epsilon,connection_epsilon
        self.failed_prefills=[];self.build_lock=threading.RLock()

    cleanup=FloatingModel.cleanup_failed_prefills

    def cache_bytes_per_slot(self,capacity):return sum(block.cache_bytes_per_slot(capacity) for block in self.blocks)

    def _input(self,token_ids,hidden,activations):
        if (not isinstance(token_ids,torch.Tensor) or token_ids.dtype!=torch.int64 or token_ids.device.type!='cpu'
                or token_ids.ndim!=2 or not 1<=token_ids.shape[0]<=128 or token_ids.shape[1]<1
                or bool(((token_ids<0)|(token_ids>=self.vocab_size)).any())):
            raise ValueError('DeepSeek stage requires checked CPU vocabulary IDs')
        batch,length=token_ids.shape
        required=4*batch*length*self.streams*self.dim
        if self.last and not activations:required=max(required,4*batch*self.vocab_size)
        if required>self.max_output_bytes:raise ValueError('DeepSeek stage boundary exceeds output budget')
        if self.first and not activations:
            if hidden is not None:raise ValueError('first DeepSeek stage embeds its original tokens')
        elif (not _float(hidden,4) or tuple(hidden.shape)!=(batch,length,self.streams,self.dim)
                or not torch.isfinite(hidden).all()):
            raise ValueError('DeepSeek stage requires finite CPU FP32 full residual streams matching token IDs')

    def _hidden(self,tokens,hidden,activations):
        return (F.embedding(tokens,self.embedding).unsqueeze(2).expand(-1,-1,self.streams,-1)
                if self.first and not activations else hidden)

    def prefill(self,token_ids,*,hidden=None,max_tokens=None,max_cache_bytes=256*1024**2,query_chunk=16):
        return self._prefill(token_ids,hidden,max_tokens,max_cache_bytes,query_chunk,False)

    def prefill_activations(self,hidden,token_ids,*,max_tokens=None,max_cache_bytes=256*1024**2,query_chunk=16):
        return self._prefill(token_ids,hidden,max_tokens,max_cache_bytes,query_chunk,True)

    @torch.no_grad()
    def _prefill(self,tokens,hidden,capacity,budget,query_chunk,activations):
        self._input(tokens,hidden,activations);length=tokens.shape[1]
        capacity=length if capacity is None else capacity
        if (type(capacity) is not int or not length<=capacity<=2**20 or type(budget) is not int or not 1<=budget<=2**31
                or type(query_chunk) is not int or not 1<=query_chunk<=128):raise ValueError('invalid DeepSeek stage prefill controls')
        if tokens.shape[0]*self.cache_bytes_per_slot(capacity)>budget:raise ValueError('DeepSeek aggregate stage cache budget exceeded')
        with self.build_lock:
            self.cleanup_failed_prefills();hidden=self._hidden(tokens,hidden,activations);layers=[]
            try:
                for block in self.blocks:
                    hidden,state=block.prefill(hidden,tokens,max_tokens=capacity,max_cache_bytes=budget,query_chunk=query_chunk)
                    layers.append(state)
                result=self._logits(hidden) if self.last and not activations else hidden
            except BaseException as error:
                pending=DeepSeekStageState(self,layers,length,capacity,budget)
                try:pending.close()
                except BaseException:
                    self.failed_prefills.append(pending);error.add_note('DeepSeek stage prefill cleanup remains owned for retry')
                raise
            return result,DeepSeekStageState(self,layers,length,capacity,budget)

    def decode(self,token_ids,state,*,position,hidden=None):return self._decode(token_ids,state,position,hidden,False)

    def decode_activations(self,hidden,token_ids,state,*,position):return self._decode(token_ids,state,position,hidden,True)

    @torch.no_grad()
    def _decode(self,tokens,state,position,hidden,activations):
        self._input(tokens,hidden,activations)
        if tokens.shape[1]!=1:raise ValueError('DeepSeek stage decode needs one token per slot')
        if not isinstance(state,DeepSeekStageState) or state.owner is not self:raise ValueError('DeepSeek stage cache owner differs')
        with state.lock:
            state._usable()
            if (type(position) is not int or position!=state.position or position>=state.capacity
                    or tokens.shape[0]!=state.layers[0].attention.cache.shape[0]):raise ValueError('DeepSeek stage cache cursor/shape differs')
            hidden=self._hidden(tokens,hidden,activations)
            try:
                for block,local in zip(self.blocks,state.layers):hidden=block.decode(hidden,local,tokens,position=position)
                result=self._logits(hidden) if self.last and not activations else hidden
            except BaseException:
                if any(layer.attention.position!=position or layer.attention.poisoned for layer in state.layers):state._poison()
                raise
            state.position=position+1
            return result
