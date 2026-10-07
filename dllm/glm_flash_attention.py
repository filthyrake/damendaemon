"""Explicit stateless Flash NoPE entry points into bounded GLM attention math."""
from .glm_attention import project_mla,sparse_latent_attention,_float


def project_nope(hidden,positions,weights,*,heads,head_dim,epsilon=1e-5,**budgets):
    result=project_mla(hidden,positions,weights,heads=heads,nope_dim=head_dim,rope_dim=0,eps=epsilon,**budgets)
    del result['rotary_keys']
    return result


def attend_nope(query,latents,kv_weight,indices,positions,*,value_dim,**budgets):
    if not _float(query,4) or not _float(latents,3):raise ValueError('invalid Flash NoPE attention inputs')
    return sparse_latent_attention(query,latents,latents[:,:,:0],kv_weight,indices,positions,
                                   nope_dim=query.shape[-1],value_dim=value_dim,**budgets)
