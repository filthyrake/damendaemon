"""Owned CPU Flash sparse attention with transactional latent/index histories."""
from dataclasses import dataclass,field
import threading
import torch
from .glm_attention import _float,_projection,_finite_parameter,_budgets
from .glm_flash_attention import project_nope,attend_nope
from .glm_flash_indexer import FlashIndexer
from .prepared_projection import linear


@dataclass
class FlashSparseState:
    owner: object
    latents: torch.Tensor
    keys: torch.Tensor
    gates: torch.Tensor
    capacity: int
    max_cache_bytes: int
    position: int = 0
    closed: bool = False
    lock: object = field(default_factory=threading.RLock,repr=False)

    def _usable(self):
        if self.closed or self.owner is None:raise RuntimeError('Flash sparse state is closed')

    def reorder(self,indices):
        with self.lock:
            self._usable()
            if (not isinstance(indices,torch.Tensor) or indices.dtype!=torch.int64 or indices.device.type!='cpu'
                    or indices.ndim!=1 or not 1<=indices.numel()<=128
                    or (indices<0).any() or (indices>=self.latents.shape[0]).any()):
                raise ValueError('invalid Flash sparse slot order')
            reservation=indices.numel()*self.capacity*self.owner.bytes_per_token
            temporary=indices.numel()*self.position*self.owner.bytes_per_token
            if reservation>self.max_cache_bytes or 2*temporary>self.owner.max_workspace_bytes:
                raise ValueError('Flash sparse reorder cache/workspace budget exceeded')
            replacement=[v.index_select(0,indices) for v in (self.latents,self.keys,self.gates)]
            self.latents,self.keys,self.gates=replacement

    def close(self):
        with self.lock:
            self.latents=self.keys=self.gates=None;self.owner=None;self.closed=True


class FlashSparseAttention:
    def __init__(self,weights,indexer,*,heads,head_dim,value_dim,epsilon=1e-5,
                 max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
        names={'q_a_proj.weight','q_a_layernorm.weight','q_b_proj.weight','kv_a_proj_with_mqa.weight',
               'kv_a_layernorm.weight','kv_b_proj.weight','o_proj.weight'}
        if (not isinstance(weights,dict) or set(weights)!=names or not isinstance(indexer,FlashIndexer)
                or any(type(v) is not int or not 1<=v<=4096 for v in (heads,head_dim,value_dim)) or heads>256
                or type(epsilon) not in (int,float) or not 0<epsilon<=1
                or indexer.top_k+indexer.pool_size-1>4096):
            raise ValueError('invalid Flash sparse composition/options')
        for name,value in weights.items():
            valid=_float(value,1) if 'layernorm' in name else _projection(value)
            if not valid or not _finite_parameter(value):raise ValueError('invalid Flash sparse parameter storage')
        dim=indexer.dim;qr=indexer.rank;rank=weights['kv_a_layernorm.weight'].numel()
        shapes={'q_a_proj.weight':(qr,dim),'q_a_layernorm.weight':(qr,),'q_b_proj.weight':(heads*head_dim,qr),
                'kv_a_proj_with_mqa.weight':(rank,dim),'kv_b_proj.weight':(heads*(head_dim+value_dim),rank),
                'o_proj.weight':(dim,heads*value_dim)}
        if any(weights[n].shape!=s for n,s in shapes.items()):raise ValueError('Flash sparse/indexer geometry mismatch')
        _budgets(0,0,max_output_bytes,max_workspace_bytes)
        self.weights=dict(weights);self.projections={n:w for n,w in weights.items() if n not in ('kv_b_proj.weight','o_proj.weight')}
        self.indexer=indexer;self.dim,self.rank,self.heads,self.head_dim,self.value_dim=dim,rank,heads,head_dim,value_dim
        self.epsilon=epsilon;self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes
        self.bytes_per_token=4*(rank+2*indexer.width)

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        if (type(batch) is not int or not 1<=batch<=128 or type(capacity) is not int or not 1<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31
                or batch*capacity*self.bytes_per_token>max_cache_bytes):
            raise ValueError('invalid Flash sparse cache geometry/budget')
        # Reserve the declared capacity against the cache budget, allocating only
        # published history. Temporary replacements have separate workspace bounds.
        return FlashSparseState(self,torch.empty(batch,0,self.rank,dtype=torch.float32,device='cpu'),
                                torch.empty(batch,0,self.indexer.width,dtype=torch.float32,device='cpu'),
                                torch.empty(batch,0,self.indexer.width,dtype=torch.float32,device='cpu'),capacity,max_cache_bytes)

    @torch.no_grad()
    def forward(self,hidden,state,*,position):
        if not isinstance(state,FlashSparseState):raise ValueError('invalid Flash sparse cache')
        with state.lock:
            state._usable()
            if state.owner is not self:raise ValueError('Flash sparse cache belongs to another layer')
            if (not _float(hidden,3) or hidden.shape[0]!=state.latents.shape[0] or hidden.shape[-1]!=self.dim
                    or type(position) is not int or position!=state.position or position+hidden.shape[1]>state.capacity):
                raise ValueError('invalid Flash sparse input/cursor/capacity')
            b,t,_=hidden.shape;end=position+t;h=self.heads;d=self.head_dim;v=self.value_dim;iw=self.indexer.width
            workspace=(128*b*(end*(self.rank+2*iw)+t*(self.dim+self.indexer.rank+h*(d+v)+2*iw+self.indexer.top_k+self.indexer.pool_size))
                       +16*64*(self.rank+4*h*(d+v)))
            pools=(end+self.indexer.pool_size-1)//self.indexer.pool_size
            workspace+=64*(b*end*iw+b*t*(self.dim+self.indexer.rank+self.indexer.heads*iw)
                           +b*pools*iw+self.indexer.heads*pools+pools*self.indexer.pool_size)
            _budgets(4*b*t*self.dim+8*b*t*(self.indexer.top_k+self.indexer.pool_size-1),workspace,
                     self.max_output_bytes,self.max_workspace_bytes)
            positions=torch.arange(position,end,dtype=torch.int64,device='cpu').expand(b,-1)
            budgets=dict(max_workspace_bytes=self.max_workspace_bytes,max_output_bytes=self.max_workspace_bytes)
            projected=project_nope(hidden,positions,self.projections,heads=h,head_dim=d,epsilon=self.epsilon,**budgets)
            new_keys,new_gates=self.indexer.project_keys(hidden)
            latents=torch.cat((state.latents,projected['latents']),dim=1)
            keys=torch.cat((state.keys,new_keys),dim=1);gates=torch.cat((state.gates,new_gates),dim=1)
            indices=self.indexer.select(hidden,projected['low_rank'],keys,gates,positions)
            attended=attend_nope(projected['query'],latents,self.weights['kv_b_proj.weight'],indices,positions,value_dim=v,**budgets)
            output=linear(attended.reshape(b,t,h*v),self.weights['o_proj.weight'])
            if not torch.isfinite(output).all():raise ValueError('nonfinite Flash sparse output')
            state.latents,state.keys,state.gates=latents,keys,gates
            state.position=end
            return output,indices
