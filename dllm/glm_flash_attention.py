"""Explicit stateless Flash NoPE entry points into bounded GLM attention math."""
from .glm_attention import _project_mla,_sparse_latent_attention,_float


def _project_nope(hidden,positions,weights,*,heads,head_dim,epsilon=1e-5,device='cpu',**budgets):
    result=_project_mla(hidden,positions,weights,heads=heads,nope_dim=head_dim,rope_dim=0,eps=epsilon,device=device,**budgets)
    del result['rotary_keys']
    return result


def _attend_nope(query,latents,kv_weight,indices,positions,*,value_dim,device='cpu',**budgets):
    if not _float(query,4,device) or not _float(latents,3,device):raise ValueError('invalid Flash NoPE attention inputs')
    return _sparse_latent_attention(query,latents,latents[:,:,:0],kv_weight,indices,positions,
                                   nope_dim=query.shape[-1],value_dim=value_dim,device=device,**budgets)


def project_nope(hidden,positions,weights,**options):
    return _project_nope(hidden,positions,weights,device='cpu',**options)


def attend_nope(query,latents,kv_weight,indices,positions,**options):
    return _attend_nope(query,latents,kv_weight,indices,positions,device='cpu',**options)
