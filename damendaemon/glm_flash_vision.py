"""Prepared FP32 Flash vision reference; no image processor or serving admission.

Independent equations transcribed from Transformers v5.17.0 glm5_next
(HuggingFace Team, Apache-2.0; docs/licenses/transformers-Apache-2.0.txt).
Checkpoint metadata: Z.AI GLM-5.3-Flash eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
(MIT; docs/licenses/glm53-flash-MIT.txt). No upstream runtime delegation.
"""
import torch
from torch.nn import functional as F
from .expert_ffn import activate_expert

ROOT='model.visual.'


def vision_specs(c):
    fields=('depth','hidden_size','num_heads','in_channels','patch_size','temporal_patch_size',
            'spatial_merge_size','out_hidden_size','intermediate_size','projection_intermediate_size')
    if (not isinstance(c,dict) or c.get('model_type')!='glm5_next_vision'
            or any(type(c.get(k)) is not int or not 1<=c[k]<=16384 for k in fields)
            or c['depth']>128 or c['num_heads']>256 or c['hidden_size']%c['num_heads']
            or (c['hidden_size']//c['num_heads'])%4 or c['spatial_merge_size']>16
            or c['in_channels']>16 or c['patch_size']>64 or c['temporal_patch_size']>16
            or c.get('hidden_act')!='silu' or c.get('attention_bias') is not True
            or type(c.get('attention_dropout')) not in (int,float) or c['attention_dropout']!=0
            or type(c.get('rms_norm_eps')) not in (int,float) or not 0<c['rms_norm_eps']<=1
            or type(c.get('swiglu_limit')) not in (int,float) or not 0<c['swiglu_limit']<=1e6):
        raise ValueError('unsupported Flash vision configuration')
    rope=c.get('rope_parameters')
    if rope is None:rope={'rope_type':'axial','rope_theta':10000.}
    if (not isinstance(rope,dict) or set(rope)-{'rope_type','rope_theta'} or rope.get('rope_type','axial')!='axial'
            or type(rope.get('rope_theta',10000.)) not in (int,float) or not 0<rope.get('rope_theta',10000.)<float('inf')):
        raise ValueError('unsupported Flash vision axial rotary configuration')
    d=c['hidden_size'];o=c['out_hidden_size'];p=c['patch_size'];m=c['spatial_merge_size'];hd=d//c['num_heads']
    specs={'patch_embed.proj.weight':(d,c['in_channels'],c['temporal_patch_size'],p,p),
           'patch_embed.proj.bias':(d,),'post_layernorm.weight':(d,),
           'downsample.weight':(o,d,m,m),'downsample.bias':(o,),
           'merger.proj.weight':(o,o),'merger.post_projection_norm.weight':(o,),'merger.post_projection_norm.bias':(o,)}
    for name,shape in (('gate_proj',(c['projection_intermediate_size'],o)),('up_proj',(c['projection_intermediate_size'],o)),
                       ('down_proj',(o,c['projection_intermediate_size']))):specs['merger.'+name+'.weight']=shape
    for i in range(c['depth']):
        prefix=f'blocks.{i}.'
        for name,shape in (('norm1.weight',(d,)),('norm2.weight',(d,)),('attn.q_norm.weight',(hd,)),('attn.k_norm.weight',(hd,))):specs[prefix+name]=shape
        for name,shape in (('attn.qkv',(3*d,d)),('attn.proj',(d,d)),('mlp.gate_proj',(c['intermediate_size'],d)),
                           ('mlp.up_proj',(c['intermediate_size'],d)),('mlp.down_proj',(d,c['intermediate_size']))):
            specs[prefix+name+'.weight']=shape;specs[prefix+name+'.bias']=(shape[0],)
    return specs


def audit_vision(config,names,headers=None):
    if not isinstance(config,dict) or config.get('model_type')!='glm5_next':raise ValueError('expected Flash root configuration')
    c=config.get('vision_config');specs=vision_specs(c);text=config.get('text_config',{});quant=config.get('quantization_config',{})
    excluded=quant.get('modules_to_not_convert',[]) if isinstance(quant,dict) else []
    if (not isinstance(text,dict) or text.get('hidden_size')!=c['out_hidden_size']
            or not isinstance(excluded,list) or not any(n in excluded for n in ('model.visual','visual'))):
        raise ValueError('Flash vision must match text width and remain excluded from quantization')
    names=list(names)
    if any(not isinstance(n,str) for n in names) or len(names)!=len(set(names)):raise ValueError('invalid Flash vision inventory names')
    expected={ROOT+n for n in specs};actual={n for n in names if n.startswith(ROOT)}
    if expected!=actual:raise ValueError('Flash vision tensor inventory mismatch')
    if headers is not None:
        for name,shape in specs.items():
            entry=headers.get(ROOT+name)
            if (not isinstance(entry,dict) or entry.get('dtype') not in ('F16','BF16','F32')
                    or not isinstance(entry.get('shape'),(list,tuple)) or any(type(n) is not int for n in entry['shape'])
                    or tuple(entry['shape'])!=shape):raise ValueError('Flash vision tensor shape/storage mismatch')
    return dict(vision_tensors=len(specs),weight_values_read=False,vision_execution_qualified=False)


class _FlashVision:
    """Borrow immutable CPU FP32 parameters and consume already ordered patch rows.

    Input rows flatten [channels, temporal patch, patch height, patch width].
    Grids are (frames, patch rows, patch columns), ordered frame then spatial
    merge block then row/column within that block. Frames attend independently.
    Budgets exclude caller-owned input storage and allocator/BLAS overhead.
    """
    def __init__(self,config,weights,*,max_weight_bytes=256*1024**2,max_tokens=4096,
                 max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2,device=None):
        self.device=torch.device('cpu') if device is None else torch.device(device)
        specs=vision_specs(config)
        if (type(max_tokens) is not int or not 1<=max_tokens<=65536
                or any(type(v) is not int or not 1<=v<=2**31 for v in (max_weight_bytes,max_workspace_bytes,max_output_bytes))
                or not isinstance(weights,dict) or set(weights)!=set(specs)):
            raise ValueError('invalid Flash vision parameters or budgets')
        for name,shape in specs.items():
            value=weights[name]
            if (not isinstance(value,torch.Tensor) or value.device!=self.device or value.dtype!=torch.float32
                    or tuple(value.shape)!=shape):raise ValueError('Flash vision requires exact same-device FP32 prepared tensors')
        if sum(v.numel()*4 for v in {id(v):v for v in weights.values()}.values())>max_weight_bytes:
            raise ValueError('Flash vision parameter budget exceeded')
        self.c=dict(config);self.c['rope_parameters']=dict(config.get('rope_parameters') or {})
        self.w=dict(weights);self.max_tokens=max_tokens;self.max_workspace_bytes=max_workspace_bytes;self.max_output_bytes=max_output_bytes

    def _norm(self,x,name):
        variance=x.square().mean(-1,keepdim=True)
        if not torch.isfinite(variance).all():raise ValueError('nonfinite Flash vision normalization')
        return x*torch.rsqrt(variance+self.c['rms_norm_eps'])*self.w[name]

    def _linear(self,x,name):
        value=F.linear(x,self.w[name+'.weight'],self.w.get(name+'.bias'))
        if not torch.isfinite(value).all():raise ValueError('nonfinite Flash vision projection')
        return value

    def _ffn(self,x,prefix):
        gate=self._linear(x,prefix+'gate_proj');up=self._linear(x,prefix+'up_proj')
        return self._linear(activate_expert(gate,up,None,self.c['swiglu_limit']),prefix+'down_proj')

    def _input_metadata(self,patches,grids,*,device=None):
        c=self.c;d=c['hidden_size'];m=c['spatial_merge_size'];heads=c['num_heads'];hd=d//heads
        width=c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
        if (not isinstance(grids,(list,tuple)) or not 1<=len(grids)<=256
                or any(not isinstance(g,(list,tuple)) or len(g)!=3 or any(type(v) is not int or not 1<=v<=self.max_tokens for v in g)
                       or g[1]%m or g[2]%m for g in grids)):
            raise ValueError('invalid Flash vision frame/patch grids')
        tokens=sum(t*h*w for t,h,w in grids);largest=max(h*w for _,h,w in grids);outputs=tokens//(m*m)
        if (tokens>self.max_tokens or not isinstance(patches,torch.Tensor) or patches.device!=(self.device if device is None else device) or patches.dtype!=torch.float32
                or patches.shape!=(tokens,width)):
            raise ValueError('Flash vision patch input/token budget mismatch')
        scratch=32*tokens*(width+d+c['intermediate_size']+c['out_hidden_size']+c['projection_intermediate_size'])+16*heads*largest**2
        if scratch>self.max_workspace_bytes or 4*outputs*c['out_hidden_size']>self.max_output_bytes:
            raise ValueError('Flash vision workspace/output budget exceeded')
        return tokens,outputs

    @torch.inference_mode()
    def forward(self,patches,grids):
        tokens,outputs=self._input_metadata(patches,grids)
        c=self.c;d=c['hidden_size'];m=c['spatial_merge_size'];heads=c['num_heads'];hd=d//heads
        width=c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
        if not torch.isfinite(patches).all():raise ValueError('nonfinite Flash vision patch input')
        positions=[];lengths=[]
        for t,h,w in grids:
            frame=[(r+i,s+j) for r in range(0,h,m) for s in range(0,w,m) for i in range(m) for j in range(m)]
            positions.extend(frame*t);lengths.extend([h*w]*t)
        coords=torch.tensor(positions,dtype=torch.float32,device=self.device)
        theta=c['rope_parameters'].get('rope_theta',10000.)
        inverse=theta**(-torch.arange(0,hd//2,2,dtype=torch.float32,device=self.device)/(hd//2))
        angles=(coords[:,:,None]*inverse).reshape(tokens,hd//2);angles=torch.cat((angles,angles),-1)
        cos,sin=angles.cos()[:,None,:],angles.sin()[:,None,:]
        hidden=F.linear(patches,self.w['patch_embed.proj.weight'].reshape(d,width),self.w['patch_embed.proj.bias'])
        for i in range(c['depth']):
            prefix=f'blocks.{i}.';x=self._norm(hidden,prefix+'norm1.weight')
            q,k,v=self._linear(x,prefix+'attn.qkv').reshape(tokens,3,heads,hd).unbind(1)
            q=self._norm(q,prefix+'attn.q_norm.weight');k=self._norm(k,prefix+'attn.k_norm.weight')
            def rotate(value):
                left,right=value.chunk(2,-1);return value*cos+torch.cat((-right,left),-1)*sin
            q,k=rotate(q),rotate(k);parts=[];start=0
            for length in lengths:
                stop=start+length
                query,key,value=[a[start:stop].transpose(0,1) for a in (q,k,v)]
                parts.append(F.scaled_dot_product_attention(query,key,value).transpose(0,1).reshape(length,d));start=stop
            hidden=hidden+self._linear(torch.cat(parts),prefix+'attn.proj')
            hidden=hidden+self._ffn(self._norm(hidden,prefix+'norm2.weight'),prefix+'mlp.')
        hidden=self._norm(hidden,'post_layernorm.weight')
        grouped=hidden.reshape(outputs,m,m,d).permute(0,3,1,2).reshape(outputs,-1)
        merged=F.linear(grouped,self.w['downsample.weight'].reshape(c['out_hidden_size'],-1),self.w['downsample.bias'])
        merged=self._linear(merged,'merger.proj')
        # Upstream merger uses ordinary LayerNorm's fixed default epsilon.
        if not torch.isfinite((merged-merged.mean(-1,keepdim=True)).square().mean(-1)).all():
            raise ValueError('nonfinite Flash vision merger normalization')
        merged=F.layer_norm(merged,(c['out_hidden_size'],),self.w['merger.post_projection_norm.weight'],self.w['merger.post_projection_norm.bias'],1e-5)
        result=self._ffn(F.gelu(merged),'merger.')
        if not torch.isfinite(result).all():raise ValueError('nonfinite Flash vision result')
        return result


class FlashVision(_FlashVision):
    """Explicit CPU reference; CUDA remains a separate opt-in entrypoint."""
    def __init__(self,config,weights,**options):
        super().__init__(config,weights,device=torch.device('cpu'),**options)
