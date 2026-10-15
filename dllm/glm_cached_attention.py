"""Owned FP32 GLM attention core with CPU-only public wrapper."""
from dataclasses import dataclass,field
import threading
import torch
from .glm_attention import linear
from .glm_attention import _project_mla as project_mla, _sparse_latent_attention as sparse_latent_attention,_float,_budgets,_projection,_finite_parameter
from .glm_indexer import _GlmIndexer
from .deepseek_position import frequencies


@dataclass
class GlmAttentionState:
    owner: object
    latents: torch.Tensor
    rotary_keys: torch.Tensor
    index_keys: torch.Tensor | None
    capacity: int
    max_cache_bytes: int
    bytes_per_slot: int
    position: int = 0
    poisoned: bool = False
    closed: bool = False
    lock: object = field(default_factory=threading.RLock,repr=False)

    def _usable(self):
        if self.closed or self.poisoned or self.owner is None:
            raise RuntimeError('GLM attention cache is closed or poisoned')

    def reorder(self,indices):
        with self.lock:
            self._usable()
            if (not isinstance(indices,torch.Tensor) or indices.ndim!=1 or indices.dtype!=torch.int64
                    or indices.device!=self.latents.device or not indices.numel()
                    or bool(((indices<0)|(indices>=self.latents.shape[0])).any())
                    or indices.numel()*self.bytes_per_slot>self.max_cache_bytes):
                raise ValueError('invalid GLM slot order or retained cache budget exceeded')
            # Allocate all replacements before publishing any; allocation failure
            # leaves the original cache usable. Budget covers retained storage,
            # not the temporary overlap while producing the replacement.
            latent=self.latents.index_select(0,indices)
            rotary=self.rotary_keys.index_select(0,indices)
            index=None if self.index_keys is None else self.index_keys.index_select(0,indices)
            self.latents,self.rotary_keys,self.index_keys=latent,rotary,index

    def close(self):
        with self.lock:
            self.latents=self.rotary_keys=self.index_keys=None
            self.owner=None
            self.closed=True


class _CachedGlmAttention:
    def __init__(self,weights,*,heads,nope_dim,rope_dim,value_dim,index_top_k,indexer_type='full',
                 rope_base=8000000.,epsilon=1e-5,max_output_bytes=64*1024**2,max_workspace_bytes=64*1024**2,device='cpu'):
        self.device=torch.device(device)
        projection={'q_a_proj.weight','q_a_layernorm.weight','q_b_proj.weight',
                    'kv_a_proj_with_mqa.weight','kv_a_layernorm.weight'}
        index_names={'wq_b.weight','wk.weight','k_norm.weight','k_norm.bias','weights_proj.weight'}
        names=projection|{'kv_b_proj.weight','o_proj.weight'}
        if indexer_type=='full':names|={'indexer.'+name for name in index_names}
        if indexer_type not in ('full','shared') or not isinstance(weights,dict) or set(weights)!=names:
            raise ValueError('invalid full/shared GLM attention parameter names')
        for name,value in weights.items():
            rank=1 if 'layernorm' in name or 'k_norm.' in name else 2
            valid=(_float(value,rank,self.device) if rank==1 or name=='indexer.weights_proj.weight' else _projection(value,self.device))
            if not valid or not _finite_parameter(value):
                raise ValueError('GLM attention weights require finite same-device FP32 tensors')
        if (any(type(v) is not int or not 1<=v<=4096 for v in (heads,nope_dim,rope_dim,value_dim,index_top_k))
                or heads>256 or rope_dim<2 or rope_dim%2
                or type(epsilon) not in (int,float) or not 0<epsilon<=1):
            raise ValueError('invalid GLM attention geometry')
        frequencies(rope_dim,base=rope_base,device=self.device)
        _budgets(0,0,max_output_bytes,max_workspace_bytes)
        dim=weights['q_a_proj.weight'].shape[1]
        qr=weights['q_a_layernorm.weight'].numel();kr=weights['kv_a_layernorm.weight'].numel()
        expected={'q_a_proj.weight':(qr,dim),'q_b_proj.weight':(heads*(nope_dim+rope_dim),qr),
                  'kv_a_proj_with_mqa.weight':(kr+rope_dim,dim),'kv_b_proj.weight':(heads*(nope_dim+value_dim),kr),
                  'o_proj.weight':(dim,heads*value_dim)}
        if any(weights[name].shape!=shape for name,shape in expected.items()):
            raise ValueError('GLM attention projection geometry mismatch')
        self.weights=weights.copy();self.projections={name:weights[name] for name in projection}
        self.heads,self.nope_dim,self.rope_dim,self.value_dim=heads,nope_dim,rope_dim,value_dim
        self.dim,self.rank,self.index_top_k=dim,kr,index_top_k
        self.rope_base,self.epsilon=rope_base,epsilon
        self.budgets=dict(max_output_bytes=max_output_bytes,max_workspace_bytes=max_workspace_bytes)
        self.indexer=(_GlmIndexer({name:weights['indexer.'+name] for name in index_names},rope_dim=rope_dim,
                                rope_base=rope_base,device=self.device,**self.budgets) if indexer_type=='full' else None)
        if self.indexer is not None and (self.indexer.dim!=dim or self.indexer.rank!=qr):
            raise ValueError('GLM indexer and latent attention geometry differs')

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        if (type(batch) is not int or not 1<=batch<=128 or type(capacity) is not int or not 1<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31):
            raise ValueError('invalid GLM cache capacity/budget')
        width=self.rank+self.rope_dim+(self.indexer.width if self.indexer is not None else 0)
        per_slot=4*capacity*width
        if batch*per_slot>max_cache_bytes:raise ValueError('GLM cache reservation exceeds budget')
        return GlmAttentionState(self,torch.empty(batch,capacity,self.rank,dtype=torch.float32,device=self.device),
                                 torch.empty(batch,capacity,self.rope_dim,dtype=torch.float32,device=self.device),
                                 None if self.indexer is None else torch.empty(batch,capacity,self.indexer.width,dtype=torch.float32,device=self.device),
                                 capacity,max_cache_bytes,per_slot)

    @torch.no_grad()
    def forward(self,hidden,state,*,position,previous_indices=None):
        if not isinstance(state,GlmAttentionState):raise ValueError('invalid GLM attention cache')
        with state.lock:
            state._usable()
            if state.owner is not self:raise ValueError('GLM cache belongs to another attention component')
            if (not _float(hidden,3,self.device) or hidden.shape[0]!=state.latents.shape[0] or hidden.shape[-1]!=self.dim
                    or type(position) is not int or position!=state.position
                    or position+hidden.shape[1]>state.capacity):
                raise ValueError('GLM cache position, shape or capacity mismatch')
            _budgets(4*hidden.numel(),0,**self.budgets)
            end=position+hidden.shape[1]
            if self.indexer is None:
                if (not isinstance(previous_indices,torch.Tensor) or previous_indices.dtype!=torch.int64
                        or previous_indices.device!=self.device
                        or previous_indices.shape!=(*hidden.shape[:2],min(self.index_top_k,end))):
                    raise ValueError('shared GLM attention requires matching prior full-indexer selections')
            elif previous_indices is not None:raise ValueError('full GLM indexer does not accept shared selections')
            positions=torch.arange(position,end,dtype=torch.int64,device=self.device).expand(hidden.shape[0],-1)
            projected=project_mla(hidden,positions,self.projections,heads=self.heads,nope_dim=self.nope_dim,
                                  rope_dim=self.rope_dim,rope_base=self.rope_base,eps=self.epsilon,device=self.device,**self.budgets)
            new_index=None if self.indexer is None else self.indexer.project_keys(hidden,positions)
            try:
                state.latents[:,position:end]=projected['latents']
                state.rotary_keys[:,position:end]=projected['rotary_keys']
                if self.indexer is not None:
                    state.index_keys[:,position:end]=new_index
                    selected,_=self.indexer.select(hidden,projected['low_rank'],state.index_keys[:,:end],positions,
                                                   top_k=self.index_top_k)
                else:selected=previous_indices
                attended=sparse_latent_attention(projected['query'],state.latents[:,:end],state.rotary_keys[:,:end],
                                                  self.weights['kv_b_proj.weight'],selected,positions,
                                                  nope_dim=self.nope_dim,value_dim=self.value_dim,device=self.device,**self.budgets)
                output=linear(attended.flatten(-2),self.weights['o_proj.weight'])
                if not torch.isfinite(output).all():raise ValueError('nonfinite GLM attention output')
            except BaseException:
                state.poisoned=True
                raise
            state.position=end
            return output,selected

    def prefill(self,hidden,*,capacity,max_cache_bytes=256*1024**2,previous_indices=None):
        if (not _float(hidden,3,self.device) or hidden.shape[-1]!=self.dim or not torch.isfinite(hidden).all()
                or type(capacity) is not int or hidden.shape[1]>capacity):
            raise ValueError('invalid GLM prefill input or capacity')
        state=self.new_state(hidden.shape[0],capacity,max_cache_bytes=max_cache_bytes)
        try:
            output,selected=self.forward(hidden,state,position=0,previous_indices=previous_indices)
            return output,selected,state
        except BaseException:
            state.close()
            raise


class CachedGlmAttention(_CachedGlmAttention):
    """CPU-only owned full/shared reference attention."""
    def __init__(self,weights,**options):
        super().__init__(weights,device='cpu',**options)
