"""Stateless FP32 GLM attention core with CPU-only public wrappers.

Independent equations from Transformers v5.15.0 glm_moe_dsa. No persistent cache,
model admission is supplied here. CUDA entrypoints are separate and explicit. Packed adapters expand tiles only.
"""
import torch
from .deepseek_position import frequencies,rotate_pairs
from .prepared_projection import PackedProjection,linear as _cpu_linear
from .cuda_prepared_projection import CudaPackedProjection


def linear(hidden,weight):
    return weight.linear(hidden) if isinstance(weight,CudaPackedProjection) else _cpu_linear(hidden,weight)


def _float(value,rank,device='cpu'):
    return (isinstance(value,torch.Tensor) and value.ndim==rank and min(value.shape)>0
            and value.dtype==torch.float32 and value.device==torch.device(device))


def _projection(value,device='cpu'):
    expected = CudaPackedProjection if torch.device(device).type=='cuda' else PackedProjection
    return (_float(value,2,device) or (isinstance(value,expected)
            and value.dtype==torch.float32 and value.device==torch.device(device)
            and value.format=='e4m3fn_f32'))


def _finite_parameter(value):
    # Packed values are checked through bounded decoded results, never expanded
    # or scanned as a whole matrix during admission.
    return not isinstance(value,torch.Tensor) or bool(torch.isfinite(value).all())


def _positions(positions,shape,device='cpu'):
    if (not isinstance(positions,torch.Tensor) or positions.shape!=shape
            or positions.dtype!=torch.int64 or positions.device!=torch.device(device)
            or bool(((positions<0)|(positions>=2**20)).any())):
        raise ValueError('invalid GLM logical positions')


def _budgets(output,scratch,max_output_bytes,max_workspace_bytes):
    if any(type(bound) is not int or not 1<=bound<=2**31 for bound in (max_output_bytes,max_workspace_bytes)):
        raise ValueError('invalid GLM attention tensor storage budget')
    if output>max_output_bytes or scratch>max_workspace_bytes:
        raise ValueError('GLM attention exceeds output/workspace tensor storage budget')


@torch.no_grad()
def _project_mla(hidden,positions,weights,*,heads,nope_dim,rope_dim,rope_base=8000000.,eps=1e-5,
                chunk=16,max_output_bytes=64*1024**2,max_workspace_bytes=64*1024**2,device='cpu'):
    names={'q_a_proj.weight','q_a_layernorm.weight','q_b_proj.weight',
           'kv_a_proj_with_mqa.weight','kv_a_layernorm.weight'}
    if not _float(hidden,3,device) or not isinstance(weights,dict) or set(weights)!=names:
        raise ValueError('invalid GLM latent projection inputs')
    if (type(heads) is not int or not 1<=heads<=256 or type(nope_dim) is not int or not 1<=nope_dim<=4096
            or type(rope_dim) is not int or not 0<=rope_dim<=4096 or rope_dim%2
            or type(chunk) is not int or not 1<=chunk<=128
            or type(eps) not in (int,float) or not 0<eps<=1):
        raise ValueError('invalid GLM latent projection geometry/options')
    for name,value in weights.items():
        if not (_float(value,1,device) if 'layernorm' in name else _projection(value,device)):
            raise ValueError('latent weights require same-device FP32 or explicit packed projections')
    batch,sequence,dim=hidden.shape
    qr=weights['q_a_layernorm.weight'].numel();kr=weights['kv_a_layernorm.weight'].numel()
    width=nope_dim+rope_dim
    expected={'q_a_proj.weight':(qr,dim),'q_b_proj.weight':(heads*width,qr),
              'kv_a_proj_with_mqa.weight':(kr+rope_dim,dim)}
    if any(weights[name].shape!=shape for name,shape in expected.items()):
        raise ValueError('GLM latent projection weight shape mismatch')
    _positions(positions,hidden.shape[:2],device)
    output=4*batch*sequence*(heads*width+qr+kr+rope_dim)
    scratch=4*batch*min(chunk,sequence)*(6*heads*width+4*qr+8*kr+8*rope_dim+dim)
    _budgets(output,scratch,max_output_bytes,max_workspace_bytes)
    if not torch.isfinite(hidden).all() or any(not _finite_parameter(w) for w in weights.values()):
        raise ValueError('GLM latent projection inputs must be finite')
    inverse=frequencies(rope_dim,base=rope_base,device=device) if rope_dim else None
    result={'query':torch.empty(batch,sequence,heads,width,dtype=torch.float32,device=device),
            'low_rank':torch.empty(batch,sequence,qr,dtype=torch.float32,device=device),
            'latents':torch.empty(batch,sequence,kr,dtype=torch.float32,device=device),
            'rotary_keys':torch.empty(batch,sequence,rope_dim,dtype=torch.float32,device=device)}
    def norm(value,weight):
        variance=value.square().mean(-1,keepdim=True)
        if not torch.isfinite(value).all() or not torch.isfinite(variance).all():
            raise ValueError('nonfinite GLM latent normalization intermediate')
        return value*torch.rsqrt(variance+eps)*weight
    for start in range(0,sequence,chunk):
        stop=min(start+chunk,sequence);x=hidden[:,start:stop];position=positions[:,start:stop]
        low=norm(linear(x,weights['q_a_proj.weight']),weights['q_a_layernorm.weight'])
        query=linear(low,weights['q_b_proj.weight']).unflatten(-1,(heads,width))
        if rope_dim:query[...,nope_dim:]=rotate_pairs(query[...,nope_dim:],position,inverse)
        packed=linear(x,weights['kv_a_proj_with_mqa.weight'])
        latent=norm(packed[...,:kr],weights['kv_a_layernorm.weight'])
        rotary=rotate_pairs(packed[...,kr:],position,inverse) if rope_dim else packed[...,kr:]
        for name,value in (('query',query),('low_rank',low),('latents',latent),('rotary_keys',rotary)):
            if not torch.isfinite(value).all():raise ValueError('nonfinite GLM latent projection result')
            result[name][:,start:stop]=value
    return result


@torch.no_grad()
def _sparse_latent_attention(query,latents,rotary_keys,kv_weight,indices,positions,*,nope_dim,value_dim,
                            key_chunk=64,max_output_bytes=64*1024**2,max_workspace_bytes=64*1024**2,device='cpu'):
    """Expand selected KV tiles and merge FP32 softmax statistics online.

    query [B,S,H,nope+rope], latents [B,T,rank], rotary keys [B,T,rope],
    kv_weight [H*(nope+value),rank], indices int64 [B,S,K]. Resident keys are
    indexed by absolute token position. -1 padding is ignored; each row must
    select at least one distinct causal token. Returns [B,S,H,value].
    """
    if (not _float(query,4,device) or not _float(latents,3,device) or not _projection(kv_weight,device)
            or not isinstance(rotary_keys,torch.Tensor) or rotary_keys.ndim!=3
            or rotary_keys.dtype!=torch.float32 or rotary_keys.device!=torch.device(device)):
        raise ValueError('GLM sparse attention requires same-device FP32 inputs')
    batch,sequence,heads,width=query.shape
    if (type(nope_dim) is not int or not 1<=nope_dim<=width or type(value_dim) is not int or not 1<=value_dim<=4096
            or type(key_chunk) is not int or not 1<=key_chunk<=256
            or latents.shape[0]!=batch or rotary_keys.shape!=(batch,latents.shape[1],width-nope_dim)
            or kv_weight.shape!=(heads*(nope_dim+value_dim),latents.shape[-1])
            or not isinstance(indices,torch.Tensor) or indices.ndim!=3 or indices.shape[:2]!=(batch,sequence)
            or not 1<=indices.shape[-1]<=4096 or indices.dtype!=torch.int64 or indices.device!=torch.device(device)):
        raise ValueError('GLM sparse attention geometry/selection mismatch')
    _positions(positions,query.shape[:2],device)
    tile=min(key_chunk,indices.shape[-1])
    scratch=4*(tile*(latents.shape[-1]+width-nope_dim+4*heads*(width+value_dim))
               +heads*(4*value_dim+4*tile+width))
    _budgets(4*batch*sequence*heads*value_dim,scratch,max_output_bytes,max_workspace_bytes)
    if any(not _finite_parameter(t) for t in (query,latents,rotary_keys,kv_weight)):
        raise ValueError('GLM sparse attention inputs must be finite')
    result=torch.empty(batch,sequence,heads,value_dim,dtype=torch.float32,device=device)
    for b in range(batch):
        for s in range(sequence):
            row=indices[b,s]
            if bool(((row < -1)|(row>=latents.shape[1])|(row>positions[b,s])).any()):
                raise ValueError('invalid or future GLM selected token')
            selected=row[row>=0]
            if not selected.numel() or selected.unique().numel()!=selected.numel():
                raise ValueError('GLM selection must contain distinct causal tokens')
            maximum=torch.full((heads,),-float('inf'),dtype=torch.float32,device=device)
            denominator=torch.zeros(heads,dtype=torch.float32,device=device)
            numerator=torch.zeros(heads,value_dim,dtype=torch.float32,device=device)
            for start in range(0,selected.numel(),key_chunk):
                ids=selected[start:start+key_chunk]
                expanded=linear(latents[b,ids],kv_weight).unflatten(-1,(heads,nope_dim+value_dim)).transpose(0,1)
                key=torch.cat((expanded[...,:nope_dim],rotary_keys[b,ids][None].expand(heads,-1,-1)),-1)
                values=expanded[...,nope_dim:]
                scores=torch.matmul(query[b,s,:,None,:],key.transpose(-1,-2)).squeeze(-2)*(width**-.5)
                if not torch.isfinite(expanded).all() or not torch.isfinite(scores).all():
                    raise ValueError('nonfinite GLM sparse attention intermediate')
                updated=torch.maximum(maximum,scores.max(-1).values)
                correction=(maximum-updated).exp()
                probabilities=(scores-updated[:,None]).exp()
                numerator=numerator*correction[:,None]+torch.matmul(probabilities[:,None,:],values).squeeze(-2)
                denominator=denominator*correction+probabilities.sum(-1)
                maximum=updated
            value=numerator/denominator[:,None]
            if not torch.isfinite(value).all():raise ValueError('nonfinite GLM sparse attention result')
            result[b,s]=value
    return result


def project_mla(hidden,positions,weights,**options):
    """CPU-only latent projection; CUDA entrypoint is explicitly separate."""
    return _project_mla(hidden,positions,weights,device='cpu',**options)


def sparse_latent_attention(query,latents,rotary_keys,kv_weight,indices,positions,**options):
    """CPU-only selected attention; preserve bounded reference behavior."""
    return _sparse_latent_attention(query,latents,rotary_keys,kv_weight,indices,positions,device='cpu',**options)
