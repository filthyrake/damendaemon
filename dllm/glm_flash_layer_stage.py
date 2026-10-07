"""Prepared Flash CPU layer ranges preserving hyperconnection residual streams."""
import threading
import torch
import torch.nn.functional as F
from .glm_attention import _float,_budgets
from .glm_flash_block import FlashBlock
from .glm_flash_model import FlashModel,FlashModelState,_batch


def validate_range(start,end,total):
    if any(type(value) is not int for value in (start,end,total)) or not 0<=start<end<=total<=128:
        raise ValueError('Flash stage requires a bounded nonempty contiguous layer range')


class FlashLayerStage:
    """An explicit range, not a complete FlashModel for generation admission.

    The first stage accepts tokens and expands embeddings into residual streams.
    Intermediate boundaries retain [batch, length, streams, width]. Only the last
    stage reduces streams and produces vocabulary logits. Prepared weights are
    immutable caller-owned values; cache ownership and cleanup are per range.
    """
    cleanup=FlashModel.cleanup
    new_state=FlashModel.new_state
    _cache_per_slot=FlashModel._cache_per_slot
    _release_failed=FlashModel._release_failed
    _logits=FlashModel._logits
    prefill=FlashModel.prefill

    def __init__(self,blocks,*,start,total,embedding=None,norm=None,head=None,
                 epsilon=1e-5,max_output_bytes=64*1024**2):
        if (not isinstance(blocks,(tuple,list)) or not blocks
                or any(not isinstance(block,FlashBlock) for block in blocks)
                or type(start) is not int or type(total) is not int
                or type(epsilon) not in (int,float) or not 0<epsilon<=1):
            raise ValueError('invalid prepared Flash stage')
        validate_range(start,start+len(blocks),total)
        self.start,self.end,self.total=start,start+len(blocks),total
        self.first,self.last=start==0,self.end==total
        self.dim,self.streams=blocks[0].dim,blocks[0].streams
        if any(block.dim!=self.dim or block.streams!=self.streams for block in blocks):
            raise ValueError('Flash stage block dimensions or residual stream counts disagree')
        if self.first:
            if not _float(embedding,2) or embedding.shape[1]!=self.dim:
                raise ValueError('first Flash stage requires its embedding')
        elif embedding is not None:raise ValueError('non-first Flash stage cannot own embeddings')
        if self.last:
            if (not _float(norm,1) or norm.shape!=(self.dim,) or not _float(head,2)
                    or head.shape[1]!=self.dim or (self.first and head.shape!=embedding.shape)):
                raise ValueError('last Flash stage requires final norm and vocabulary head')
        elif norm is not None or head is not None:raise ValueError('non-final Flash stage cannot own norm/head')
        if any(not torch.isfinite(value).all() for value in (embedding,norm,head) if value is not None):
            raise ValueError('Flash stage boundary weights must be finite')
        _budgets(0,0,max_output_bytes,1)
        self.embedding,self.norm,self.head=embedding,norm,head
        self.blocks=tuple(blocks);self.epsilon,self.max_output_bytes=epsilon,max_output_bytes
        self.failed_cleanups=[];self.lock=threading.RLock()

    def _tokens(self,inputs,all_logits):
        if type(all_logits) is not bool or (all_logits and not self.last):
            raise ValueError('only the final Flash stage produces all-token logits')
        if self.first:
            if (not isinstance(inputs,torch.Tensor) or inputs.ndim!=2 or min(inputs.shape)<1
                    or inputs.dtype!=torch.int64 or inputs.device.type!='cpu'
                    or bool(((inputs<0)|(inputs>=self.embedding.shape[0])).any())):
                raise ValueError('first Flash stage requires nonempty CPU vocabulary IDs')
        elif (not _float(inputs,4) or min(inputs.shape)<1 or tuple(inputs.shape[2:])!=(self.streams,self.dim)
                or not torch.isfinite(inputs).all()):
            raise ValueError('Flash stage requires finite CPU FP32 residual-stream chunks')
        batch,length=inputs.shape[:2]
        required=4*batch*length*self.streams*self.dim
        if self.last:required=max(required,4*batch*(length if all_logits else 1)*self.head.shape[0])
        if required>self.max_output_bytes:raise ValueError('Flash stage boundary exceeds output budget')

    @torch.no_grad()
    def forward(self,inputs,state,*,position,all_logits=False):
        self._tokens(inputs,all_logits)
        if not isinstance(state,FlashModelState):raise ValueError('invalid Flash stage cache')
        with state.lock:
            state._usable()
            if (state.owner is not self or type(position) is not int or position!=state.position
                    or inputs.shape[0]!=_batch(state.layers[0].attention)
                    or position+inputs.shape[1]>state.capacity):
                raise ValueError('Flash stage cache owner/position/shape mismatch')
            hidden=(F.embedding(inputs,self.embedding).unsqueeze(-2).expand(-1,-1,self.streams,-1)
                    if self.first else inputs)
            try:
                for block,layer in zip(self.blocks,state.layers):hidden=block.forward(hidden,layer,position=position)
                result=self._logits(hidden,all_logits) if self.last else hidden
            except BaseException:
                if any(layer.poisoned or layer.position!=position
                       or layer.attention.position!=position for layer in state.layers):state._poison()
                raise
            state.position=position+inputs.shape[1]
            return result
