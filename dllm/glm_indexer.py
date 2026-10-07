"""Bounded CPU FP32 GLM-5.3 full-indexer reference equations.

Independent composition from Transformers v5.15.0 glm_moe_dsa. This is not
FP8 execution, a persistent cache, a complete attention layer or a GPU kernel.
"""
import torch
import torch.nn.functional as F
from .compressed_indexer import select_compressed
from .deepseek_position import frequencies,rotate_pairs


def _tensor(value,rank):
    return (isinstance(value,torch.Tensor) and value.ndim==rank and value.device.type=='cpu'
            and value.dtype==torch.float32 and min(value.shape)>0)


class GlmIndexer:
    """Read-only FP32 weights with explicit output and tiled arithmetic budgets."""
    def __init__(self,weights,*,rope_dim,rope_base=8000000.,max_output_bytes=64*1024**2,
                 max_workspace_bytes=64*1024**2):
        names={'wq_b.weight','wk.weight','k_norm.weight','k_norm.bias','weights_proj.weight'}
        if not isinstance(weights,dict) or set(weights)!=names:
            raise ValueError('GLM indexer requires exactly its five projection/norm tensors')
        for name,value in weights.items():
            if not _tensor(value,1 if name.startswith('k_norm.') else 2):
                raise ValueError('GLM indexer weights require nonempty CPU FP32 tensors')
        width,dim=weights['wk.weight'].shape
        heads=weights['weights_proj.weight'].shape[0]
        rank=weights['wq_b.weight'].shape[1]
        if (not 1<=heads<=256 or not 2<=width<=4096
                or weights['weights_proj.weight'].shape!=(heads,dim)
                or weights['wq_b.weight'].shape!=(heads*width,rank)
                or weights['k_norm.weight'].shape!=(width,) or weights['k_norm.bias'].shape!=(width,)
                or type(rope_dim) is not int or not 2<=rope_dim<=width or rope_dim%2):
            raise ValueError('GLM indexer projection geometry mismatch')
        for bound in (max_output_bytes,max_workspace_bytes):
            if type(bound) is not int or not 1<=bound<=2**31:
                raise ValueError('invalid GLM indexer tensor storage budget')
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ValueError('GLM indexer weights must be finite')
        self.weights=weights.copy()
        self.width,self.dim,self.heads,self.rank=width,dim,heads,rank
        self.rope_dim=rope_dim
        self.inverse=frequencies(rope_dim,base=rope_base)
        self.max_output_bytes,self.max_workspace_bytes=max_output_bytes,max_workspace_bytes

    def _inputs(self,hidden,positions,chunk):
        if (not _tensor(hidden,3) or hidden.shape[-1]!=self.dim or not torch.isfinite(hidden).all()
                or not isinstance(positions,torch.Tensor) or positions.dtype!=torch.int64
                or positions.device.type!='cpu' or positions.shape!=hidden.shape[:2]
                or bool(((positions<0)|(positions>=2**20)).any())
                or type(chunk) is not int or not 1<=chunk<=128):
            raise ValueError('invalid GLM indexer hidden/position/chunk inputs')
        return hidden.shape[:2]

    def _budget(self,output,workspace):
        if output>self.max_output_bytes or workspace>self.max_workspace_bytes:
            raise ValueError('GLM indexer output/workspace exceeds configured tensor storage budget')

    @torch.no_grad()
    def project_keys(self,hidden,positions,*,chunk=16):
        batch,sequence=self._inputs(hidden,positions,chunk)
        rows=min(chunk,sequence)
        self._budget(4*batch*sequence*self.width,4*batch*rows*(self.dim+12*self.width))
        keys=torch.empty(batch,sequence,self.width,dtype=torch.float32,device='cpu')
        for start in range(0,sequence,chunk):
            stop=min(start+chunk,sequence)
            projected=F.linear(hidden[:,start:stop],self.weights['wk.weight'])
            normalized=F.layer_norm(projected,(self.width,),self.weights['k_norm.weight'],
                                    self.weights['k_norm.bias'],eps=1e-6)
            normalized[...,:self.rope_dim]=rotate_pairs(normalized[...,:self.rope_dim],
                                                        positions[:,start:stop],self.inverse)
            keys[:,start:stop]=normalized
        return keys

    @torch.no_grad()
    def select(self,hidden,low_rank,keys,positions,*,top_k,query_chunk=16,key_chunk=64):
        batch,sequence=self._inputs(hidden,positions,query_chunk)
        if (not _tensor(low_rank,3) or low_rank.shape!=(*hidden.shape[:2],self.rank)
                or not _tensor(keys,3) or keys.shape[0]!=batch or keys.shape[-1]!=self.width
                or not torch.isfinite(low_rank).all() or not torch.isfinite(keys).all()
                or type(top_k) is not int or not 1<=top_k<=4096
                or type(key_chunk) is not int or not 1<=key_chunk<=256):
            raise ValueError('invalid GLM indexer query/key/selection inputs')
        top=min(top_k,keys.shape[1]);q=min(query_chunk,sequence);k=min(key_chunk,keys.shape[1])
        # Conservative explicit tensor scratch allowance: projections/rotation,
        # score tiles, merged candidates, IDs and stable-sort intermediates.
        scratch=4*batch*(q*self.heads*(12*self.width+4*k)+q*(self.dim+self.rank)
                         +k*self.width+12*q*(top+k))
        self._budget(12*batch*sequence*top,scratch)
        selected=torch.empty(batch,sequence,top,dtype=torch.int64,device='cpu')
        scores=torch.empty(batch,sequence,top,dtype=torch.float32,device='cpu')
        for start in range(0,sequence,query_chunk):
            stop=min(start+query_chunk,sequence)
            queries=F.linear(low_rank[:,start:stop],self.weights['wq_b.weight']).unflatten(-1,(self.heads,self.width))
            queries[...,:self.rope_dim]=rotate_pairs(queries[...,:self.rope_dim],positions[:,start:stop],self.inverse)
            weights=F.linear(hidden[:,start:stop],self.weights['weights_proj.weight'])*(self.heads**-.5)
            ids,values=select_compressed(queries,keys,weights,positions[:,start:stop],ratio=1,top_k=top_k,
                                        query_chunk=query_chunk,key_chunk=key_chunk,
                                        scale_head_weights=False,dot_scale=self.width**-.5)
            selected[:,start:stop],scores[:,start:stop]=ids,values
        return selected,scores
