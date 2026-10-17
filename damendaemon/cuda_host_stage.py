"""Bounded synchronous host payload boundaries around explicitly owned CUDA slots."""
import torch
from .cuda_stage_slots import CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots


def _host_tensor(value,dtype,shape):
    if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=dtype
            or value.ndim!=len(shape) or any(expected is not None and actual!=expected for actual,expected in zip(value.shape,shape))
            or min(value.shape)<1):
        raise ValueError('host CUDA boundary requires matching CPU payload geometry and dtype')
    return value.numel()*value.element_size()


def _budget(required,limit):
    if type(limit) is not int or not 1<=limit<=2**31 or required>limit:
        raise ValueError('host CUDA boundary transfer payload budget exceeded')


class CudaHostStageSlots:
    """Own a fresh slot set, keeping compute_device truthful and boundaries CPU.

    max_transfer_bytes bounds the sum of logical input/output payload bytes per
    call, not allocator/cache/scratch memory. Copies are blocking, unpinned and
    contiguous; no streams, peer access, transport or device selection is implied.
    Callers must not mutate input tensors or use the claimed slots independently.
    """
    boundary_device=torch.device('cpu')
    compute_dtype=torch.float32

    def __init__(self,slots,*,max_transfer_bytes):
        if not isinstance(slots,(CudaGlmStageSlots,CudaFlashStageSlots,CudaDeepSeekStageSlots)):
            raise ValueError('host boundary requires explicit CUDA slots')
        _budget(0,max_transfer_bytes)
        with slots.lock:
            slots._ready()
            if (getattr(slots,'_boundary_owner',None) is not None or getattr(slots,'_pipeline_owner',None) is not None
                    or any(slots.lengths) or any(state is not None for state in slots.states)):
                raise ValueError('host boundary requires fresh unclaimed CUDA slots')
            self.slots=slots;self.max_transfer_bytes=max_transfer_bytes;self.lock=slots.lock
            slots._boundary_owner=object()

    @property
    def actual_device(self):return self.slots.actual_device
    @property
    def stage(self):return self.slots.stage
    @property
    def c(self):return self.slots.c
    @property
    def layer_start(self):return self.slots.layer_start
    @property
    def layer_end(self):return self.slots.layer_end
    @property
    def context(self):return self.slots.context
    @property
    def max_batch(self):return self.slots.max_batch
    @property
    def states(self):return self.slots.states
    @property
    def lengths(self):return self.slots.lengths
    @property
    def closed(self):return self.slots.closed
    @property
    def failed(self):return self.slots.failed
    @property
    def reserved_cache_bytes(self):return self.slots.reserved_cache_bytes
    @property
    def length(self):return self.slots.length
    @length.setter
    def length(self,value):self.slots.length=value

    def _to_device(self,value):return value.to(self.actual_device,copy=True).contiguous()
    def _to_host(self,value):return value.to('cpu',copy=True).contiguous()

    def _copy(self,value,*,out=False):
        try:return self._to_host(value) if out else self._to_device(value)
        except BaseException as error:
            self.slots._fail(error);raise

    def _tokens(self,tokens,batch=None,length=None):
        size=_host_tensor(tokens,torch.int64,(batch,length))
        if tokens.shape[0]>self.max_batch or tokens.shape[1]>self.context:
            raise ValueError('host CUDA token extent exceeds slots/context')
        return size

    def _hidden(self,hidden,*,output=False):
        shape=(None,*self.slots._hidden_shape) if output else (None,None,*self.slots._hidden_shape)
        size=_host_tensor(hidden,torch.float32,shape)
        if hidden.shape[0]>self.max_batch or (not output and hidden.shape[1]>self.context):
            raise ValueError('host CUDA activation extent exceeds slots/context')
        if size>self.stage.max_output_bytes:raise ValueError('host CUDA activation output budget exceeded')
        return size

    @torch.no_grad()
    def embedding(self,tokens):
        with self.lock:
            self.slots._ready()
            if not self.stage.first:raise ValueError('only the first CUDA stage embeds')
            size=self._tokens(tokens);width=1
            for dimension in self.slots._hidden_shape:width*=dimension
            output=tokens.numel()*width*4
            _budget(size+output,self.max_transfer_bytes)
            if output>self.stage.max_output_bytes:raise ValueError('host CUDA embedding output budget exceeded')
            return self._copy(self.slots.embedding(self._copy(tokens)),out=True)

    @torch.no_grad()
    def text_embeddings(self,tokens):
        """Collapsed Flash rows for CPU image insertion; no residual frame copy."""
        with self.lock:
            self.slots._ready()
            if not isinstance(self.slots,CudaFlashStageSlots) or not self.stage.first:
                raise ValueError('collapsed embedding rows require the first CUDA Flash stage')
            size=self._tokens(tokens);output=tokens.numel()*self.stage.dim*4
            _budget(size+output,self.max_transfer_bytes)
            if output>self.stage.max_output_bytes:raise ValueError('host Flash embedding row budget exceeded')
            # Flash embedding expands identical residual streams as a view. Copy
            # one stream only, preserving the original single embedding lookup.
            values=self.slots.embedding(self._copy(tokens))[:,:,0]
            return self._copy(values,out=True)

    @torch.no_grad()
    def forward_layers(self,hidden,positions,slots,prefill,prefill_start=0,*,token_ids=None):
        with self.lock:
            self.slots._ready();size=self._hidden(hidden)
            routing=isinstance(self.slots,CudaDeepSeekStageSlots)
            if routing:size+=self._tokens(token_ids,*hidden.shape[:2])
            elif token_ids is not None:raise ValueError('this CUDA stage does not accept routing IDs')
            _budget(size+hidden.numel()*4,self.max_transfer_bytes)
            if not torch.isfinite(hidden).all():raise ValueError('host CUDA activations must be finite')
            result=self.slots.forward_layers(self._copy(hidden),positions,slots,prefill,prefill_start,
                token_ids=self._copy(token_ids) if routing else None)
            return self._copy(result,out=True)

    @torch.no_grad()
    def output_logits(self,hidden):
        with self.lock:
            self.slots._ready()
            if not self.stage.last:raise ValueError('only the final CUDA stage projects logits')
            size=self._hidden(hidden,output=True);output=hidden.shape[0]*self.slots._vocab_size*4
            _budget(size+output,self.max_transfer_bytes)
            if output>self.stage.max_output_bytes:raise ValueError('host CUDA logits output budget exceeded')
            if not torch.isfinite(hidden).all():raise ValueError('host CUDA activations must be finite')
            return self._copy(self.slots.output_logits(self._copy(hidden)),out=True)

    def release(self,slots):return self.slots.release(slots)
    def close(self):return self.slots.close()
