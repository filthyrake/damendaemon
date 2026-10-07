"""Bounded CPU FP32 Flash learned-pool indexer, independently transcribed.

Reference: Transformers v5.17.0 glm5_next. Contiguous valid-token intervals only;
left/right padding supported, internal holes rejected. Ties choose earlier pools.
Returned int64 indices reserve top_k slots for complete pools and pool_size-1
slots for the tail, with -1 padding. Slot layout is canonical rather than the
reference's variable-length tail placement; selected token sets are equivalent.
"""
import torch
from torch.nn import functional as F
from .glm_flash_kda import _tensor,_finite


class FlashIndexer:
    def __init__(self,weights,*,heads,pool_size=4,top_k=2048,
                 max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
        if (not isinstance(weights,dict) or not isinstance(weights.get('wk.weight'),torch.Tensor)
                or weights['wk.weight'].ndim!=2
                or any(type(v) is not int or not 1<=v<=4096 for v in (heads,pool_size,top_k))
                or top_k%pool_size or top_k<pool_size
                or any(type(v) is not int or not 1<=v<=2**31 for v in (max_workspace_bytes,max_output_bytes))):
            raise ValueError('invalid Flash indexer geometry/budgets')
        width,dim=weights['wk.weight'].shape
        query=weights.get('wq_b.weight')
        if not isinstance(query,torch.Tensor) or query.ndim!=2:raise ValueError('invalid Flash indexer query projection')
        rank=query.shape[1]
        shapes={'wk.weight':(width,dim),'wq_b.weight':(heads*width,rank),
                'weights_proj.weight':(heads,dim),'k_norm.weight':(width,),'k_norm.bias':(width,),
                'index_kpool_compress_ape':(pool_size,width),'index_kpool_compress_gate':(width,dim)}
        if (not 1<=width<=4096 or not 1<=dim<=2**21 or not 1<=rank<=2**21
                or set(weights)!=set(shapes) or any(not _tensor(weights[n],s) for n,s in shapes.items())):
            raise ValueError('invalid Flash indexer weight shapes/storage')
        _finite(*weights.values())
        self.weights=dict(weights);self.dim,self.rank,self.width,self.heads=dim,rank,width,heads
        self.pool_size,self.top_k=pool_size,top_k
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes

    def _budget(self,workspace,output):
        if workspace>self.max_workspace_bytes or output>self.max_output_bytes:
            raise ValueError('Flash indexer workspace/output budget exceeded')

    @torch.no_grad()
    def project_keys(self,hidden):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or min(hidden.shape[:2])<1
                or not _tensor(hidden,(*hidden.shape[:2],self.dim))):
            raise ValueError('invalid Flash indexer hidden states')
        rows=hidden.shape[0]*hidden.shape[1]
        self._budget(64*rows*(self.dim+self.width),8*rows*self.width)
        _finite(hidden,*self.weights.values());w=self.weights
        raw=F.linear(hidden,w['wk.weight']);_finite(raw)
        centered=raw-raw.mean(-1,keepdim=True);variance=centered.square().mean(-1,keepdim=True);_finite(variance)
        keys=centered*torch.rsqrt(variance+1e-6)*w['k_norm.weight']+w['k_norm.bias']
        gates=F.linear(hidden,w['index_kpool_compress_gate']);_finite(keys,gates)
        return keys,gates

    @torch.no_grad()
    def select(self,hidden,low_rank,keys,gates,positions,*,valid_keys=None):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=3 or min(hidden.shape[:2])<1
                or not _tensor(hidden,(*hidden.shape[:2],self.dim))):
            raise ValueError('invalid Flash indexer query states')
        b,t,_=hidden.shape;d=self.width;p=self.pool_size
        if (not _tensor(low_rank,(b,t,self.rank)) or not isinstance(keys,torch.Tensor) or keys.ndim!=3
                or not keys.shape[1] or not _tensor(keys,(b,keys.shape[1],d)) or not _tensor(gates,keys.shape)
                or not isinstance(positions,torch.Tensor) or positions.device.type!='cpu' or positions.dtype!=torch.int64
                or positions.shape!=(b,t)):
            raise ValueError('invalid Flash indexer cache/query geometry')
        length=keys.shape[1];pools=(length+p-1)//p
        self._budget(64*(b*length*d+b*t*(self.dim+self.rank+self.heads*d)+b*pools*d+self.heads*pools+pools*p),
                     8*b*t*(self.top_k+p-1))
        if (positions<0).any() or (positions>=length).any():raise ValueError('invalid Flash indexer positions')
        if valid_keys is None:valid_keys=torch.ones((b,length),dtype=torch.bool,device='cpu')
        if (not isinstance(valid_keys,torch.Tensor) or valid_keys.device.type!='cpu'
                or valid_keys.dtype!=torch.bool or valid_keys.shape!=(b,length)):
            raise ValueError('invalid Flash indexer padding mask')
        _finite(hidden,low_rank,keys,gates,*self.weights.values())
        q=F.linear(low_rank,self.weights['wq_b.weight']).reshape(b,t,self.heads,d)
        head_weights=F.linear(hidden,self.weights['weights_proj.weight'])/(self.heads**.5)
        _finite(q,head_weights)
        result=torch.full((b,t,self.top_k+p-1),-1,dtype=torch.int64,device='cpu')
        for batch in range(b):
            real=valid_keys[batch].nonzero().flatten()
            if not real.numel():continue
            first,last=int(real[0]),int(real[-1])
            if last-first+1!=real.numel():raise ValueError('Flash indexer does not support internal padding holes')
            count=real.numel()//p
            if count:
                indices=torch.arange(first,first+count*p,dtype=torch.int64,device='cpu').reshape(count,p)
                grouped=keys[batch,indices];logits=gates[batch,indices]+self.weights['index_kpool_compress_ape']
                _finite(logits)
                pooled=(logits.softmax(1)*grouped).sum(1);_finite(pooled)
            for token in range(t):
                pos=int(positions[batch,token])
                if not valid_keys[batch,pos]:continue
                visible=pos-first+1;complete=visible//p
                if complete:
                    # Only complete visible pools enter scoring; no future pool
                    # contributes even when the cache contains a full prefill.
                    scores=(q[batch,token]@pooled[:complete].T/(d**.5)).relu()
                    scores=(scores*head_weights[batch,token,:,None]).sum(0);_finite(scores)
                    selected=torch.argsort(scores,descending=True,stable=True)[:self.top_k//p]
                    raw=indices[selected].flatten();result[batch,token,:raw.numel()]=raw
                tail=visible%p
                if tail:result[batch,token,self.top_k:self.top_k+tail]=torch.arange(pos-tail+1,pos+1,dtype=torch.int64,device='cpu')
        return result
