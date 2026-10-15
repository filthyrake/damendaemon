"""Owned FP32 GLM Flash KDA core with a CPU-only public layer."""
from dataclasses import dataclass,field
import threading
import torch
from torch.nn import functional as F
from .glm_flash_kda import _recurrent_attention as recurrent_attention,_tensor,_finite


@dataclass
class FlashLinearState:
    owner: object
    convolution: torch.Tensor
    recurrent: torch.Tensor
    capacity: int
    max_cache_bytes: int
    position: int = 0
    closed: bool = False
    lock: object = field(default_factory=threading.RLock,repr=False)

    def _usable(self):
        if self.closed or self.owner is None:raise RuntimeError('Flash KDA state is closed')

    def reorder(self,indices):
        with self.lock:
            self._usable()
            if (not isinstance(indices,torch.Tensor) or indices.dtype!=torch.int64 or indices.device!=self.recurrent.device
                    or indices.ndim!=1 or not 1<=indices.numel()<=128
                    or (indices<0).any() or (indices>=self.recurrent.shape[0]).any()):
                raise ValueError('invalid Flash KDA slot order')
            retained=indices.numel()*self.owner.bytes_per_slot
            if retained>self.max_cache_bytes or 2*retained>self.owner.max_workspace_bytes:
                raise ValueError('Flash KDA reorder cache/workspace budget exceeded')
            convolution=self.convolution.index_select(0,indices)
            recurrent=self.recurrent.index_select(0,indices)
            self.convolution,self.recurrent=convolution,recurrent

    def close(self):
        with self.lock:
            self.convolution=self.recurrent=None
            self.owner=None;self.closed=True


class _FlashLinearAttention:
    def __init__(self,weights,*,heads,head_dim,kernel_size=4,lower_bound=-5.,epsilon=1e-5,
                 max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2,device='cpu'):
        self.device=torch.device(device)
        if (any(type(v) is not int or not 1<=v<=4096 for v in (heads,head_dim,kernel_size))
                or type(lower_bound) not in (int,float) or not -100<=lower_bound<0
                or type(epsilon) not in (int,float) or not 0<epsilon<=1
                or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))
                or not isinstance(weights,dict) or not isinstance(weights.get('q_proj.weight'),torch.Tensor)
                or weights['q_proj.weight'].ndim!=2):
            raise ValueError('invalid Flash KDA layer options')
        dim=weights['q_proj.weight'].shape[1];width=heads*head_dim
        if not 1<=dim<=2**21:raise ValueError('invalid Flash KDA hidden width')
        shapes={f'{n}_proj.weight':(width,dim) for n in ('q','k','v')}
        shapes.update({f'{n}_conv1d.weight':(width,1,kernel_size) for n in ('q','k','v')})
        shapes.update({f'{n}_a_proj.weight':(head_dim,dim) for n in ('f','g')})
        shapes.update({f'{n}_b_proj.weight':(width,head_dim) for n in ('f','g')})
        shapes.update({'b_proj.weight':(heads,dim),'o_proj.weight':(dim,width),'o_norm.weight':(head_dim,),
                       'A_log':(heads,),'dt_bias':(width,)})
        if set(weights)!=set(shapes) or any(not _tensor(weights[n],shape,self.device) for n,shape in shapes.items()):
            raise ValueError('invalid Flash KDA weight shapes/storage')
        _finite(*weights.values())
        self.weights=dict(weights);self.dim,self.heads,self.head_dim,self.kernel_size=dim,heads,head_dim,kernel_size
        self.lower_bound,self.epsilon=lower_bound,epsilon
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes
        self.bytes_per_slot=4*(3*width*(kernel_size-1)+heads*head_dim*head_dim)

    def new_state(self,batch,capacity,*,max_cache_bytes=256*1024**2):
        if (type(batch) is not int or not 1<=batch<=128 or type(capacity) is not int or not 1<=capacity<=2**20
                or type(max_cache_bytes) is not int or not 1<=max_cache_bytes<=2**31
                or batch*self.bytes_per_slot>max_cache_bytes):
            raise ValueError('invalid Flash KDA cache geometry/budget')
        return FlashLinearState(self,torch.zeros(batch,3,self.heads*self.head_dim,self.kernel_size-1,dtype=torch.float32,device=self.device),
                                torch.zeros(batch,self.heads,self.head_dim,self.head_dim,dtype=torch.float32,device=self.device),
                                capacity,max_cache_bytes)

    @torch.no_grad()
    def forward(self,hidden,state,*,position):
        if not isinstance(state,FlashLinearState):raise ValueError('invalid Flash KDA state')
        with state.lock:
            state._usable()
            if state.owner is not self:raise ValueError('Flash KDA state belongs to another layer')
            if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3
                    or not _tensor(hidden,(state.recurrent.shape[0],hidden.shape[1],self.dim),self.device)
                    or not hidden.shape[1] or type(position) is not int or position!=state.position
                    or position+hidden.shape[1]>state.capacity):
                raise ValueError('invalid Flash KDA input/cursor/capacity')
            b,t,_=hidden.shape;h,d=self.heads,self.head_dim;w=h*d
            # Includes all projected sequences, causal tails, recurrence scratch,
            # temporary validation masks, and replacements until atomic publish.
            workspace=256*b*t*(self.dim+8*w)+256*b*h*d*d+16*b*3*w*self.kernel_size
            if workspace>self.max_workspace_bytes or 4*b*t*self.dim>self.max_output_bytes:
                raise ValueError('Flash KDA layer workspace/output budget exceeded')
            _finite(hidden,*self.weights.values(),state.convolution,state.recurrent)
            weights=self.weights
            project=lambda x,name:F.linear(x,weights[name+'.weight'])
            convolved=[];tails=[]
            for i,name in enumerate(('q','k','v')):
                projected=project(hidden,name+'_proj').transpose(1,2)
                _finite(projected)
                joined=torch.cat((state.convolution[:,i],projected),dim=-1)
                tails.append(joined[:,:,-(self.kernel_size-1):].clone() if self.kernel_size>1 else joined[:,:,:0].clone())
                convolved.append(F.silu(F.conv1d(joined,weights[name+'_conv1d.weight'],groups=w)).transpose(1,2).reshape(b,t,h,d))
            forget=project(project(hidden,'f_a_proj'),'f_b_proj')+weights['dt_bias']
            rate=weights['A_log'].exp();_finite(forget,rate)
            gate_input=forget.reshape(b,t,h,d)*rate.reshape(1,1,h,1);_finite(gate_input)
            log_decay=self.lower_bound*gate_input.sigmoid()
            beta_logits=project(hidden,'b_proj');_finite(beta_logits)
            beta=beta_logits.sigmoid()
            core,recurrent=recurrent_attention(*convolved,log_decay,beta,state.recurrent,
                                               max_workspace_bytes=self.max_workspace_bytes,
                                               max_output_bytes=self.max_workspace_bytes,device=self.device)
            gate=project(project(hidden,'g_a_proj'),'g_b_proj').reshape(b,t,h,d)
            variance=core.square().mean(-1,keepdim=True);_finite(gate,variance)
            normalized=core*torch.rsqrt(variance+self.epsilon)*weights['o_norm.weight']*gate.sigmoid()
            output=project(normalized.reshape(b,t,w),'o_proj')
            convolution=torch.stack(tails,dim=1)
            _finite(output,convolution)
            # Every operation above is functional against the previous cache.
            # Failed calls leave both histories and the cursor unchanged.
            state.convolution,state.recurrent=convolution,recurrent
            state.position+=t
            return output


class FlashLinearAttention(_FlashLinearAttention):
    """CPU-only prepared Flash linear-attention layer."""
    def __init__(self,weights,**options):
        super().__init__(weights,device='cpu',**options)
