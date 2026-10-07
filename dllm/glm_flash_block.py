"""CPU FP32 Flash multi-stream residual blocks with partial-failure ownership."""
from dataclasses import dataclass
import torch
from .glm_attention import _float,_budgets
from .glm_flash_hyper import FlashHyperConnection
from .glm_flash_linear import FlashLinearAttention
from .glm_flash_sparse import FlashSparseAttention
from .glm_flash_ffn import FlashFeedForward,PackedFlashFeedForward


@dataclass
class FlashBlockState:
    owner: object
    attention: object
    position: int = 0
    poisoned: bool = False

    def _usable(self):
        if self.owner is None or self.attention.closed or self.poisoned:
            raise RuntimeError('Flash block state is closed or poisoned')
        if self.attention.owner is not self.owner.attention or self.attention.position!=self.position:
            self.poisoned=True
            raise RuntimeError('Flash block attention ownership/cursors disagree')

    def reorder(self,indices):
        with self.attention.lock:
            self._usable();self.attention.reorder(indices)

    def close(self):
        with self.attention.lock:
            self.attention.close();self.owner=None


class FlashBlock:
    def __init__(self,attention,feed_forward,attention_hyper,ffn_hyper,input_norm,post_norm,*,epsilon=1e-5,
                 max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
        if (not isinstance(attention,(FlashLinearAttention,FlashSparseAttention))
                or not isinstance(feed_forward,(FlashFeedForward,PackedFlashFeedForward))
                or attention.dim!=feed_forward.dim
                or any(not isinstance(h,FlashHyperConnection) or h.width!=attention.dim for h in (attention_hyper,ffn_hyper))
                or attention_hyper.streams!=ffn_hyper.streams
                or any(not _float(w,1) or w.shape!=(attention.dim,) or not torch.isfinite(w).all() for w in (input_norm,post_norm))
                or type(epsilon) not in (int,float) or not 0<epsilon<=1):
            raise ValueError('invalid Flash residual block composition')
        _budgets(0,0,max_output_bytes,max_workspace_bytes)
        self.attention,self.feed_forward=attention,feed_forward
        self.attention_hyper,self.ffn_hyper=attention_hyper,ffn_hyper
        self.input_norm,self.post_norm,self.epsilon=input_norm,post_norm,epsilon
        self.dim,self.streams=attention.dim,attention_hyper.streams
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        return FlashBlockState(self,self.attention.new_state(batch,capacity,max_cache_bytes=max_cache_bytes))

    @torch.no_grad()
    def forward(self,hidden,state,*,position):
        if not isinstance(state,FlashBlockState) or state.owner is not self:
            raise ValueError('Flash block state belongs to another component')
        with state.attention.lock:
            state._usable()
            if (not _float(hidden,4) or hidden.shape[-2:]!=(self.streams,self.dim)
                    or type(position) is not int or position!=state.position):
                raise ValueError('invalid Flash block input/cursor')
            # Residual temporaries are additional to component scratch budgets.
            _budgets(4*hidden.numel(),64*hidden.numel(),self.max_output_bytes,self.max_workspace_bytes)
            def norm(value,weight):
                variance=value.square().mean(-1,keepdim=True)
                if not torch.isfinite(variance).all() or not torch.isfinite(weight).all():
                    raise ValueError('nonfinite Flash block normalization')
                return value*torch.rsqrt(variance+self.epsilon)*weight
            before=state.attention.position
            try:
                collapsed,post,matrix=self.attention_hyper.reduce(hidden)
                attended=self.attention.forward(norm(collapsed,self.input_norm),state.attention,position=position)
                if isinstance(self.attention,FlashSparseAttention):attended=attended[0]
                residual=self.attention_hyper.restore(attended,hidden,post,matrix)
                collapsed,post,matrix=self.ffn_hyper.reduce(residual)
                fed=self.feed_forward(norm(collapsed,self.post_norm))
                output=self.ffn_hyper.restore(fed,residual,post,matrix)
            except BaseException:
                if state.attention.position!=before:state.poisoned=True
                raise
            state.position=state.attention.position
            return output
