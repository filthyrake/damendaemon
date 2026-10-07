"""Explicit prepared CPU hybrid reference stages, including canonical Q4/Q8.

This constructor never probes/selects accelerators or widens their admission.
Packed weights/scales are borrowed immutably; floating tensors become FP32.
The caller owns source storage. Stage budgets cover retained stage tensors,
conversion/rotary scratch, caches and bounded projection workspace separately.
"""
import copy
import math
import torch
from torch.nn import functional as F
from .hybrid_stage import HybridLayerStage
from .hybrid_layout import stage_weight_names,hybrid_state_bytes
from .prepared_projection import PackedProjection
from .packed_tiles import validate_layout
from .packed_projection import projection_plan


def _layout(model,start,end):
    c=model.config;mode=model.quantization
    integers=('hidden_size','vocab_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim',
              'linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim',
              'linear_conv_kernel_dim','intermediate_size','rotary_dim','max_position_embeddings')
    if (not isinstance(c,dict) or c.get('model_type')!='qwen3_5_text' or mode not in ('none','q4_0','q8_0')
            or getattr(model,'vision_weights',None) is not None or not isinstance(model.weights,dict) or not isinstance(model.scales,dict)
            or any(type(c.get(k)) is not int or not 1<=c[k]<=2**21 for k in integers)
            or c['num_hidden_layers']>256 or c['num_attention_heads']%c['num_key_value_heads']
            or c['linear_num_value_heads']%c['linear_num_key_heads'] or c['rotary_dim']%2 or c['rotary_dim']>c['head_dim']
            or not isinstance(c.get('layer_types'),list) or len(c['layer_types'])!=c['num_hidden_layers']
            or any(k not in ('full_attention','linear_attention') for k in c['layer_types'])
            or type(c.get('rms_norm_eps')) not in (int,float) or not 0<c['rms_norm_eps']<=1
            or type(c.get('rope_theta')) not in (int,float) or not 0<c['rope_theta']<float('inf')
            or any(not isinstance(n,str) for n in model.weights) or set(model.scales)-set(model.weights)):
        raise ValueError('invalid prepared CPU hybrid configuration/storage')
    names=stage_weight_names(model.weights,c['num_hidden_layers'],start,end)
    d=c['hidden_size'];h=c['num_attention_heads'];kh=c['num_key_value_heads'];hd=c['head_dim']
    lk=c['linear_num_key_heads'];lv=c['linear_num_value_heads'];dk=c['linear_key_head_dim'];dv=c['linear_value_head_dim']
    expected={}
    if start==0:expected['embed']=(c['vocab_size'],d)
    if end==c['num_hidden_layers']:expected.update(norm=(d,),head=(c['vocab_size'],d))
    for i in range(start,end):
        expected.update({f'{i}.norm1':(d,),f'{i}.norm2':(d,),f'{i}.gate_up':(2*c['intermediate_size'],d),f'{i}.down':(d,c['intermediate_size'])})
        if c['layer_types'][i]=='full_attention':
            shapes={'qkv':(2*(h+kh)*hd,d),'o':(d,h*hd),'q_norm':(hd,),'k_norm':(hd,)}
        else:
            cw=2*lk*dk+lv*dv
            shapes={'qkv':(cw,d),'z':(lv*dv,d),'b':(lv,d),'a':(lv,d),'out':(d,lv*dv),
                    'conv':(cw,1,c['linear_conv_kernel_dim']),'dt':(lv,),'A':(lv,),'gated_norm':(dv,)}
        expected.update({f'{i}.{n}':shape for n,shape in shapes.items()})
    if set(names)!=set(expected):raise ValueError('CPU hybrid stage tensor inventory mismatch')
    selected={n:model.weights[n] for n in names};scales={n:model.scales[n] for n in names if n in model.scales}
    format={'q4_0':'q4_0_f16','q8_0':'q8_0_f16'}.get(mode);bindings={}
    for name,weight in selected.items():
        if not isinstance(weight,torch.Tensor) or weight.device.type!='cpu':raise ValueError('CPU stage requires CPU source storage')
        if name in scales:
            if format is None or name=='embed' or len(expected[name])!=2:raise ValueError('unsupported CPU hybrid packed tensor')
            if validate_layout(weight,scales[name],format=format)!=expected[name]:raise ValueError('CPU hybrid packed geometry mismatch')
            if id(weight) in bindings and bindings[id(weight)] is not scales[name]:raise ValueError('aliased packed weight has ambiguous scales')
            bindings[id(weight)]=scales[name]
        elif weight.dtype not in (torch.float16,torch.bfloat16,torch.float32) or tuple(weight.shape)!=expected[name]:
            raise ValueError('CPU hybrid floating geometry/storage mismatch or missing scales')
    return selected,scales,format


class CpuStageCapacityError(ValueError):
    """Valid prepared geometry exceeds the caller's allocation limits."""


def cpu_stage_budget(model,start,end,max_batch=4,*,max_weight_bytes=256*1024**2,max_cache_bytes=256*1024**2,
                     projection_workspace_bytes=8*1024**2,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Validate storage metadata and report conservative peak bytes without allocating tensors."""
    weights,scales,format=_layout(model,start,end);c=model.config
    bounds=(max_weight_bytes,max_cache_bytes,projection_workspace_bytes,max_workspace_bytes,max_output_bytes)
    if (any(type(v) is not int or not 1<=v<=2**31 for v in bounds)
            or type(max_batch) is not int or not 1<=max_batch<=4
            or type(model.max_context) is not int or not 1<=model.max_context<=min(2**20,c['max_position_embeddings'])):
        raise ValueError('invalid CPU hybrid stage capacity/budgets')
    unique={id(v):v for v in weights.values()}
    weight_bytes=sum(v.numel()*(4 if v.is_floating_point() else v.element_size()) for v in unique.values())
    weight_bytes+=sum(v.numel()*v.element_size() for v in {id(v):v for v in scales.values()}.values())
    largest=max((v.numel()*v.element_size() for v in unique.values() if v.dtype in (torch.float16,torch.bfloat16)),default=0)
    owned=dict(c,num_hidden_layers=end-start,layer_types=c['layer_types'][start:end])
    cache_bytes=hybrid_state_bytes(owned,model.max_context,max_batch,4)
    rotary_bytes=8*model.max_context*(c['rotary_dim']+1)
    if (weight_bytes+largest>max_weight_bytes or cache_bytes+rotary_bytes>max_cache_bytes
            or 32*model.max_context*c['rotary_dim']>max_workspace_bytes
            or projection_workspace_bytes>max_workspace_bytes):
        raise CpuStageCapacityError('CPU hybrid stage weight/cache/workspace budget exceeded')
    for name,scale in scales.items():
        try:projection_plan(weights[name],scale,format=format,row_chunk=64,column_chunk=256,
                            workspace_bytes=projection_workspace_bytes)
        except ValueError as error:raise CpuStageCapacityError('CPU packed projection workspace budget exceeded') from error
    return dict(layer_start=start,layer_end=end,weight_bytes=weight_bytes,conversion_overlap_bytes=largest,
                state_bytes=cache_bytes,rotary_bytes=rotary_bytes,workspace_bytes=max_workspace_bytes,
                output_bytes=max_output_bytes,
                required_memory_bytes=weight_bytes+largest+cache_bytes+rotary_bytes+max_workspace_bytes+max_output_bytes)


class CpuHybridStage(HybridLayerStage):
    def __init__(self,model,start,end,max_batch=4,*,max_weight_bytes=256*1024**2,max_cache_bytes=256*1024**2,
                 projection_workspace_bytes=8*1024**2,max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
        weights,scales,format=_layout(model,start,end);c=model.config
        budget=cpu_stage_budget(model,start,end,max_batch,max_weight_bytes=max_weight_bytes,max_cache_bytes=max_cache_bytes,
                                projection_workspace_bytes=projection_workspace_bytes,max_workspace_bytes=max_workspace_bytes,
                                max_output_bytes=max_output_bytes)
        unique={id(v):v for v in weights.values()}
        weight_bytes=budget['weight_bytes']
        # Construct adapters (metadata only) before any promotion/cache allocation.
        adapters={id(weights[n]):PackedProjection(weights[n],scale,format=format,workspace_bytes=projection_workspace_bytes)
                  for n,scale in scales.items()}
        self.closed=False;self._failed=False
        identity=getattr(model,'_hybrid_stage_identity',None)
        if identity is None:identity=model._hybrid_stage_identity=object()
        self.source_identity=identity;self.model=copy.copy(model);self.model.config=dict(c)
        self.model.weights={};self.model.scales={}
        self.c=self.model.config
        promoted={id(v):v.float() if v.is_floating_point() else v for v in unique.values()}
        self.w={n:promoted[id(v)] for n,v in weights.items()}
        self.model.weights=dict(self.w);self.model.scales=dict(scales);self.model.bytes=weight_bytes
        self.adapters=adapters;self.linear=self._linear
        self.context,self.max_batch=model.max_context,max_batch;self.lengths=[0]*max_batch;self.length=0
        self.layer_start,self.layer_end=start,end;self.first=start==0;self.last=end==c['num_hidden_layers']
        self.device='cpu';self.compute_dtype=self.residual_dtype=torch.float32;self.hybrid_precision='cpu_reference'
        self.kernel_backend='cpu-reference-'+model.quantization;self.fused=self.metal_scan=self.chunk_scan=self.offload=None
        self.eps=c['rms_norm_eps'];self.heads,self.kh,self.dim=c['num_attention_heads'],c['num_key_value_heads'],c['head_dim']
        self.lk,self.lv=c['linear_num_key_heads'],c['linear_num_value_heads'];self.dk,self.dv=c['linear_key_head_dim'],c['linear_value_head_dim']
        self.conv_width=2*self.lk*self.dk+self.lv*self.dv
        self.max_workspace_bytes,self.max_output_bytes=max_workspace_bytes,max_output_bytes
        self.required_memory_bytes=budget['required_memory_bytes']
        self.stage_activation_workspace_bytes=max_workspace_bytes
        self.output_head_workspace_bytes=max_output_bytes if self.last else 0
        self.kv={};self.conv={};self.state={};self.batch_graphs={}
        for i in range(start,end):
            if c['layer_types'][i]=='full_attention':self.kv[i]=torch.zeros(2,max_batch,self.kh,self.context,self.dim,dtype=torch.float32,device='cpu')
            else:
                self.conv[i]=torch.zeros(max_batch,self.conv_width,c['linear_conv_kernel_dim'],dtype=torch.float32,device='cpu')
                self.state[i]=torch.zeros(max_batch,self.lv,self.dk,self.dv,dtype=torch.float32,device='cpu')
        rd=c['rotary_dim'];inverse=c['rope_theta']**(-torch.arange(0,rd,2,dtype=torch.float32,device='cpu')/rd)
        angles=torch.outer(torch.arange(self.context,device='cpu'),inverse);angles=torch.cat((angles,angles),-1)
        self.cos,self.sin=angles.cos(),angles.sin();self.arange=torch.arange(self.context,device='cpu')
        if not torch.isfinite(self.cos).all() or not torch.isfinite(self.sin).all():raise ValueError('nonfinite CPU hybrid rotary table')

    @property
    def failed(self):return self._failed

    @property
    def supports_chunked_prefill(self):return True

    def _usable(self):
        if self.closed or self._failed:raise RuntimeError('CPU hybrid stage is closed or failed')

    def _linear(self,x,weight,bias=None):
        adapter=self.adapters.get(id(weight));rows=adapter.shape[0] if adapter else weight.shape[0]
        if (not isinstance(x,torch.Tensor) or x.device.type!='cpu' or x.dtype!=torch.float32
                or x.ndim<2 or 4*math.prod(x.shape[:-1])*rows>self.max_output_bytes):
            raise ValueError('CPU hybrid projection output/input budget mismatch')
        if not torch.isfinite(x).all():raise ValueError('nonfinite CPU hybrid projection input')
        if adapter is not None:
            if bias is not None:raise ValueError('packed CPU hybrid bias is unsupported')
            result=adapter.linear(x)
        else:result=F.linear(x,weight,bias)
        if not torch.isfinite(result).all():raise ValueError('nonfinite CPU hybrid projection output')
        return result

    def norm(self,x,weight,dtype=None):
        variance=x.square().mean(-1,keepdim=True)
        if not torch.isfinite(variance).all():raise ValueError('nonfinite CPU hybrid normalization variance')
        return x*torch.rsqrt(variance+self.eps)*weight

    def attention(self,x,i,positions,slots,prefill,prefix_length=0):
        n,length,_=x.shape;w=self.w
        q,k,v=self.linear(x,w[f'{i}.qkv']).split((2*self.heads*self.dim,self.kh*self.dim,self.kh*self.dim),-1)
        q,gate=q.reshape(n,length,self.heads,2*self.dim).chunk(2,-1)
        k,v=[a.reshape(n,length,self.kh,self.dim) for a in (k,v)]
        q=self.rotate(self.norm(q,w[f'{i}.q_norm']),positions);k=self.rotate(self.norm(k,w[f'{i}.k_norm']),positions)
        ck,cv=self.kv[i].unbind(0);ck[slots[:,None],:,positions,:]=k;cv[slots[:,None],:,positions,:]=v
        mask=self.arange[None,None,None,:]<=positions[:,None,:,None]
        out=F.scaled_dot_product_attention(q.transpose(1,2),ck.index_select(0,slots),cv.index_select(0,slots),attn_mask=mask,enable_gqa=True)
        out=out.transpose(1,2)*gate.sigmoid()
        return self.linear(out.reshape(n,length,-1),w[f'{i}.o'])

    @torch.inference_mode()
    def forward_layers(self,x,positions,slots,prefill,prefix_length=0,layer_start=None,layer_end=None):
        self._usable();start=self.layer_start if layer_start is None else layer_start;end=self.layer_end if layer_end is None else layer_end
        if (not isinstance(x,torch.Tensor) or x.ndim!=3 or x.dtype!=torch.float32 or x.device.type!='cpu'
                or not 1<=x.shape[0]<=self.max_batch or not 1<=x.shape[1]<=self.context or x.shape[2]!=self.c['hidden_size']
                or type(start) is not int or type(end) is not int or not self.layer_start<=start<end<=self.layer_end
                or type(prefill) is not bool or type(prefix_length) is not int or not 0<=prefix_length<self.context
                or not isinstance(positions,torch.Tensor) or positions.dtype!=torch.int64 or positions.device.type!='cpu' or positions.shape!=x.shape[:2]
                or not isinstance(slots,torch.Tensor) or slots.dtype!=torch.int64 or slots.device.type!='cpu' or slots.shape!=(x.shape[0],)
                or (positions<0).any() or (positions>=self.context).any() or (slots<0).any() or (slots>=self.max_batch).any()
                or slots.unique().numel()!=slots.numel() or (not prefill and (x.shape[1]!=1 or prefix_length))):
            raise ValueError('invalid CPU hybrid execution geometry')
        if prefill and not torch.equal(positions,torch.arange(prefix_length,prefix_length+x.shape[1],device='cpu').expand(x.shape[0],-1)):
            raise ValueError('CPU hybrid prefill requires contiguous shared positions')
        b,t,_=x.shape
        scratch=32*b*self.lv*self.dk*self.dv+8*b*self.kh*self.context*self.dim+64*b*t*(self.c['hidden_size']+self.c['intermediate_size']+self.conv_width+self.heads*self.dim+self.lv*self.dv)+16*b*self.heads*t*self.context
        if scratch>self.max_workspace_bytes or 4*x.numel()>self.max_output_bytes:raise ValueError('CPU hybrid execution budget exceeded')
        try:
            for i in range(start,end):
                hidden=self.norm(x,self.w[f'{i}.norm1'])
                x=x+(self.linear_attention(hidden,i,slots,prefill,prefix_length) if self.c['layer_types'][i]=='linear_attention'
                     else self.attention(hidden,i,positions,slots,prefill,prefix_length))
                gate,up=self.linear(self.norm(x,self.w[f'{i}.norm2']),self.w[f'{i}.gate_up']).chunk(2,-1)
                x=x+self.linear(F.silu(gate)*up,self.w[f'{i}.down'])
            if not torch.isfinite(x).all():raise ValueError('nonfinite CPU hybrid residual output')
            return x
        except BaseException:self._failed=True;raise

    def embedding(self,ids):
        self._usable()
        if (not isinstance(ids,torch.Tensor) or ids.device.type!='cpu' or ids.dtype!=torch.int64
                or ids.ndim!=2 or not 1<=ids.shape[0]<=self.max_batch or not 1<=ids.shape[1]<=self.context
                or 4*ids.numel()*self.c['hidden_size']>self.max_output_bytes
                or (ids<0).any() or (ids>=self.c['vocab_size']).any()):
            raise ValueError('CPU hybrid embedding output budget exceeded')
        return super().embedding(ids)

    def output_logits(self,hidden):
        self._usable()
        try:
            if not self.last:raise ValueError('only the last CPU hybrid stage owns the output head')
            if (not isinstance(hidden,torch.Tensor) or hidden.device.type!='cpu' or hidden.dtype!=torch.float32
                    or hidden.ndim!=2 or not 1<=hidden.shape[0]<=self.max_batch or hidden.shape[1]!=self.c['hidden_size']
                    or 16*hidden.numel()>self.max_workspace_bytes
                    or 4*hidden.shape[0]*self.c['vocab_size']>self.max_output_bytes):
                raise ValueError('invalid CPU hybrid output geometry/budget')
            return self.linear(self.norm(hidden,self.w['norm']),self.w['head'])
        except BaseException:self._failed=True;raise

    def close(self):
        if self.closed:return
        super().close();self.adapters.clear();self.model.scales.clear()

    def release_partial(self):
        super().release_partial()
        getattr(self,'adapters',{}).clear()
        if getattr(self,'model',None) is not None:self.model.scales.clear()
