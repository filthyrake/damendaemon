"""FP32 GLM residual block core with CPU-only public composition."""
from dataclasses import dataclass
import torch
from .glm_attention import _float
from .glm_cached_attention import CachedGlmAttention,GlmAttentionState
from .glm_ffn import GlmFeedForward
from .glm_normalization import rms_norm


@dataclass
class GlmBlockState:
    owner: object
    attention: GlmAttentionState
    position: int = 0
    poisoned: bool = False

    def reorder(self,indices):
        with self.attention.lock:
            if self.poisoned:raise RuntimeError('GLM block cache is poisoned')
            self.attention.reorder(indices)

    def close(self):
        with self.attention.lock:
            self.attention.close()
            self.owner=None


class _GlmBlock:
    def __init__(self,attention,feed_forward,input_norm,post_norm,*,epsilon=1e-5,device='cpu',attention_type=CachedGlmAttention,ffn_type=GlmFeedForward):
        self.device=torch.device(device)
        if (not isinstance(attention,attention_type) or not isinstance(feed_forward,ffn_type)
                or attention.device!=self.device or getattr(feed_forward,'device',torch.device('cpu'))!=self.device
                or attention.dim!=feed_forward.dim
                or any(not _float(w,1,self.device) or w.shape!=(attention.dim,) or not torch.isfinite(w).all() for w in (input_norm,post_norm))
                or type(epsilon) not in (int,float) or not 0<epsilon<=1):
            raise ValueError('invalid GLM block composition')
        self.attention,self.feed_forward=attention,feed_forward
        self.input_norm,self.post_norm,self.epsilon=input_norm,post_norm,epsilon
        self.dim=attention.dim

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        return GlmBlockState(self,self.attention.new_state(batch,capacity,max_cache_bytes=max_cache_bytes))

    @torch.no_grad()
    def forward(self,hidden,state,*,position,previous_indices=None):
        if not isinstance(state,GlmBlockState) or state.owner is not self:
            raise ValueError('GLM block cache belongs to another component')
        with state.attention.lock:
            if state.poisoned:raise RuntimeError('GLM block cache is poisoned')
            if state.position!=state.attention.position:
                state.poisoned=state.attention.poisoned=True
                raise RuntimeError('GLM block and attention cursors disagree')
            if not _float(hidden,3,self.device) or hidden.shape[-1]!=self.dim or type(position) is not int or position!=state.position:
                raise ValueError('invalid GLM block input/position')
            def norm(value,weight):return rms_norm(value,weight,self.epsilon)
            before=state.attention.position
            try:
                attended,indices=self.attention.forward(norm(hidden,self.input_norm),state.attention,
                                                         position=position,previous_indices=previous_indices)
                residual=hidden+attended
                output=residual+self.feed_forward(norm(residual,self.post_norm))
                if not torch.isfinite(output).all():raise ValueError('nonfinite GLM block output')
            except BaseException:
                if state.attention.poisoned or state.attention.position!=before:
                    state.poisoned=state.attention.poisoned=True
                raise
            state.position=state.attention.position
            return output,indices

    def prefill(self,hidden,*,capacity,max_cache_bytes=256*1024**2,previous_indices=None):
        if not _float(hidden,3,self.device) or hidden.shape[-1]!=self.dim:raise ValueError('invalid GLM block prefill')
        state=self.new_state(hidden.shape[0],capacity,max_cache_bytes=max_cache_bytes)
        try:
            output,indices=self.forward(hidden,state,position=0,previous_indices=previous_indices)
            return output,indices,state
        except BaseException:
            state.close()
            raise


class GlmBlock(_GlmBlock):
    """CPU-only residual block composition."""
    def __init__(self,attention,feed_forward,input_norm,post_norm,*,epsilon=1e-5):
        super().__init__(attention,feed_forward,input_norm,post_norm,epsilon=epsilon,device='cpu')
