"""Independent packed GPT-OSS execution; Harmony serving is a separate gate."""
import os

import torch
from torch.nn import functional as F

from .batch import BatchExecutor
from .device import selected_device


def validate_config(config):
    from .config import decoder_config, _positive_number
    c=dict(config)
    scaling=c.get('rope_scaling')
    if (not isinstance(scaling,dict) or scaling.get('rope_type')!='yarn'
            or scaling.get('truncate') is not False):
        raise ValueError('GPT-OSS requires the untruncated YaRN schedule')
    quant=c.get('quantization_config')
    exclusions=['model.layers.*.self_attn','model.layers.*.mlp.router','model.embed_tokens','lm_head']
    if (not isinstance(quant,dict) or quant.get('quant_method')!='mxfp4'
            or set(quant)-{'quant_method','modules_to_not_convert'}
            or quant.get('modules_to_not_convert')!=exclusions):
        raise ValueError('GPT-OSS requires checkpoint-native MXFP4 experts and unquantized dense projections')
    if c.get('attention_bias') is not True or c.get('tie_word_embeddings',False):
        raise ValueError('GPT-OSS requires biased attention and an independent output head')
    # Reuse the established dimension/numerical checks without permitting the
    # generic dense loader to interpret expert tensors as ordinary matrices.
    common=decoder_config({**c,'model_type':'llama','attention_bias':False,
        'quantization_config':None,'layer_types':[],'sliding_window':None})
    for name in ('num_local_experts','num_experts_per_tok','sliding_window'):
        if type(c.get(name)) is not int or c[name]<=0:
            raise ValueError(f'{name} must be a positive integer')
    if c['num_experts_per_tok']>c['num_local_experts']:
        raise ValueError('expert count per token exceeds the expert bank')
    if c.get('experts_per_token',c['num_experts_per_tok'])!=c['num_experts_per_tok']:
        raise ValueError('conflicting expert counts')
    if c['hidden_size']%32 or c['intermediate_size']%32:
        raise ValueError('MXFP4 projection widths must be multiples of 32')
    layers=c.get('layer_types')
    if (not isinstance(layers,list) or len(layers)!=c['num_hidden_layers']
            or any(x not in ('full_attention','sliding_attention') for x in layers)):
        raise ValueError('GPT-OSS needs one full/sliding attention type per layer')
    limit=c.get('swiglu_limit',7.)
    if not _positive_number(limit):
        raise ValueError('SwiGLU limit must be finite and positive')
    return {**common,**c,'swiglu_limit':limit}


def load_weights(state,c):
    hidden=c['hidden_size'];middle=c['intermediate_size'];experts=c['num_local_experts']
    q=c['num_attention_heads']*c['head_dim'];kv=c['num_key_value_heads']*c['head_dim']
    def take(name,shape,matrix=False,packed=False):
        value=state.pop(name)
        if tuple(value.shape)!=tuple(shape):raise ValueError(f'incorrect GPT-OSS tensor shape: {name}')
        if packed:
            if value.dtype!=torch.uint8:raise ValueError(f'expected packed bytes: {name}')
            if name.endswith('_scales') and (value==255).any():
                raise ValueError(f'reserved MXFP4 scale in checkpoint: {name}')
            return value.contiguous()
        if value.dtype not in (torch.bfloat16,torch.float16,torch.float32) or not torch.isfinite(value).all():
            raise ValueError(f'invalid floating tensor: {name}')
        if matrix:
            half=value.half()
            # Preserve every checkpoint value. FP16 storage is used only when
            # conversion is lossless; unusual values retain their source dtype.
            return (half if torch.equal(half.float(),value.float()) else value).contiguous()
        return value.float().contiguous()
    w={'embed':take('model.embed_tokens.weight',(c['vocab_size'],hidden),matrix=True),
       'head':take('lm_head.weight',(c['vocab_size'],hidden),matrix=True),
       'norm':take('model.norm.weight',(hidden,))}
    for i in range(c['num_hidden_layers']):
        root=f'model.layers.{i}.'
        matrices=[];biases=[]
        for p,n in (('q',q),('k',kv),('v',kv)):
            matrices.append(take(root+f'self_attn.{p}_proj.weight',(n,hidden),matrix=True))
            biases.append(take(root+f'self_attn.{p}_proj.bias',(n,)))
        w[f'{i}.qkv.weight']=torch.cat(matrices)
        w[f'{i}.qkv.bias']=torch.cat(biases)
        for dst,src,shape,matrix in (
            ('out','self_attn.o_proj.weight',(hidden,q),True),
            ('out.bias','self_attn.o_proj.bias',(hidden,),False),
            ('sinks','self_attn.sinks',(c['num_attention_heads'],),False),
            ('norm1','input_layernorm.weight',(hidden,),False),
            ('norm2','post_attention_layernorm.weight',(hidden,),False),
            ('router','mlp.router.weight',(experts,hidden),True),
            ('router.bias','mlp.router.bias',(experts,),False)):
            w[f'{i}.{dst}']=take(root+src,shape,matrix=matrix)
        for dst,src,n,k in (('up','gate_up_proj',2*middle,hidden),('down','down_proj',hidden,middle)):
            prefix=root+'mlp.experts.'+src
            w[f'{i}.{dst}.blocks']=take(prefix+'_blocks',(experts,n,k//32,16),packed=True)
            w[f'{i}.{dst}.scales']=take(prefix+'_scales',(experts,n,k//32),packed=True)
            w[f'{i}.{dst}.bias']=take(prefix+'_bias',(experts,n))
    if state:raise ValueError('unexpected GPT-OSS checkpoint tensors: '+', '.join(sorted(state)[:4]))
    return w


class GptOssExecutor(BatchExecutor):
    def __init__(self,model,use_graph=True,max_batch=1,kernels=None):
        if selected_device()!='cuda' or (kernels or os.environ.get('DLLM_KERNELS','auto'))=='torch':
            raise ValueError('GPT-OSS requires native CUDA kernels')
        from . import mxfp4_kernels,moe_kernels
        from .mxfp4 import routed_linear,scale_capabilities
        self.grouped_prefill=os.environ.get('DLLM_GPT_OSS_PREFILL','grouped')
        if self.grouped_prefill not in ('routed','grouped'):
            raise ValueError('DLLM_GPT_OSS_PREFILL must be routed or grouped')
        self.expert_half={};self.expert_normal={}
        for i in range(model.config['num_hidden_layers']):
            for projection in ('up','down'):
                scales=model.weights[f'{i}.{projection}.scales']
                # Every possible E2M1 code is exactly representable in FP16
                # within this scale range, including FP16 subnormals.
                self.expert_half[i,projection],self.expert_normal[i,projection]=scale_capabilities(scales)
        self.chunked_experts=True
        self.normal_experts=True
        self.sorted_prefill_routes=True
        self.prefill_attention_bytes=32*1024**2
        attention_mode=os.environ.get('DLLM_GPT_OSS_ATTENTION','fused')
        if attention_mode not in ('native','fused'):
            raise ValueError('DLLM_GPT_OSS_ATTENTION must be native or fused')
        self.fused_prefill_attention=attention_mode=='fused'
        self.attention_projection_policy='ieee_fp32_prefill_256_plus'
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('GPT-OSS attention projections require torch FP32 matmul precision highest')
        self.expert_linear=routed_linear
        self.expert_activation=mxfp4_kernels.swiglu
        self.combine=moe_kernels.combine
        super().__init__(model,use_graph,max_batch,kernels)
        self.kernel_backend+='-mxfp4'

    def experts(self,x,i,prefill=False):
        w=self.w
        router=self.dense_linear(x,w[f'{i}.router'],w[f'{i}.router.bias'])
        # GPT-OSS normalizes only the selected logits, after top-k selection.
        logits,selected=router.topk(self.c['num_experts_per_tok'],dim=-1)
        probabilities=logits.softmax(-1)
        if prefill and self.grouped_prefill=='grouped' and x.shape[0]>=16:
            return self.grouped_experts(x,i,selected,probabilities)
        up=self.expert_linear(x,w[f'{i}.up.blocks'],w[f'{i}.up.scales'],selected,w[f'{i}.up.bias'],chunked=self.chunked_experts,
            normal_scales=self.normal_experts and self.expert_normal[i,'up'])
        activation=self.expert_activation(up,self.c['swiglu_limit'])
        down=self.expert_linear(activation,w[f'{i}.down.blocks'],w[f'{i}.down.scales'],selected,
            w[f'{i}.down.bias'],False,chunked=self.chunked_experts,
            normal_scales=self.normal_experts and self.expert_normal[i,'down'])
        return self.combine(down,probabilities)

    def grouped_experts(self,x,i,selected,probabilities):
        from .mxfp4 import unpack
        w=self.w;top=selected.shape[1]
        projected=torch.empty((selected.numel(),self.c['hidden_size']),device=x.device,dtype=x.dtype)
        if self.sorted_prefill_routes:
            from .expert_routes import grouped_routes
            groups=grouped_routes(selected,self.c['num_local_experts'])
        else:
            def original_groups():
                for expert in range(self.c['num_local_experts']):
                    tokens,routes=torch.where(selected==expert)
                    if tokens.numel():yield expert,tokens,tokens*top+routes
            groups=original_groups()
        for expert,tokens,indices in groups:
            hidden=x.index_select(0,tokens)
            # A bounded set of matrix shapes avoids compiling a new kernel
            # for every expert's live token count. Padding rows cannot affect
            # real rows; discard their biased outputs before route scatter.
            count=tokens.numel();bucket=1<<(count-1).bit_length()
            if bucket!=count:hidden=F.pad(hidden,(0,0,0,bucket-count))
            weight=unpack(w[f'{i}.up.blocks'][expert],w[f'{i}.up.scales'][expert],
                torch.float16 if self.expert_half[i,'up'] else torch.float32)
            up=self.dense_linear(hidden,weight)
            up.add_(w[f'{i}.up.bias'][expert])
            del weight,hidden
            activation=self.expert_activation(up,self.c['swiglu_limit'])
            del up
            weight=unpack(w[f'{i}.down.blocks'][expert],w[f'{i}.down.scales'][expert],
                torch.float16 if self.expert_half[i,'down'] else torch.float32)
            output=self.dense_linear(activation,weight)
            output.add_(w[f'{i}.down.bias'][expert])
            del weight,activation
            projected.index_copy_(0,indices,output[:count])
            del output
        return self.combine(projected,probabilities)

    @torch.inference_mode()
    def forward(self,ids,positions,prefill,slot=0):
        return self._forward(ids,positions,torch.full_like(positions,slot),prefill,slot if prefill else None)

    @torch.inference_mode()
    def forward_batch(self,ids,positions,slots):
        return self._forward(ids,positions,slots,False)

    def attention_projection(self,x,weight,bias,prefill):
        if prefill and x.shape[0]>=256:
            if torch.backends.cuda.matmul.allow_tf32:
                raise ValueError('GPT-OSS attention projections require torch FP32 matmul precision highest')
            # Preserve FP32 attention products at longer prefixes. Compensated
            # FP16 products can cross nearly tied expert-routing boundaries.
            # Expand one matrix transiently; retaining FP32 copies would not fit A2.
            return F.linear(x.float(),weight.float(),bias.float())
        return self.dense_linear(x,weight,bias)

    def layer(self,x,positions,slots,i,prefill=False,prefill_slot=None):
        w=self.w
        qkv=self.attention_projection(self.norm(x,w[f'{i}.norm1']),w[f'{i}.qkv.weight'],w[f'{i}.qkv.bias'],prefill)
        ck,cv=self.kv[i].unbind(0)
        query=self.kernels.rope_batch(qkv,self.cos,self.sin,positions,slots,ck,cv,self.heads,precise=True)
        window=self.c['sliding_window'] if self.c['layer_types'][i]=='sliding_attention' else 0
        if prefill and self.fused_prefill_attention and prefill_slot is not None and x.shape[0]>=256:
            from .sink_attention import prefill_attention
            count=x.shape[0]
            attended=prefill_attention(query,ck[prefill_slot,:,:count],cv[prefill_slot,:,:count],w[f'{i}.sinks'],window)
        else:
            attended=self.kernels.attention(query,ck,cv,positions,slots,window=window,sinks=w[f'{i}.sinks'],
                scratch_bytes=self.prefill_attention_bytes if prefill else None)
        x=x+self.attention_projection(attended.reshape(x.shape[0],-1),w[f'{i}.out'],w[f'{i}.out.bias'],prefill)
        # Expert expansion must not overlap dead long-prompt attention tensors.
        del qkv,query,attended
        return x+self.experts(self.norm(x,w[f'{i}.norm2']),i,prefill)

    def _forward(self,ids,positions,slots,prefill,prefill_slot=None):
        x=F.embedding(ids,self.w['embed']).float()
        for i in range(self.c['num_hidden_layers']):x=self.layer(x,positions,slots,i,prefill,prefill_slot=prefill_slot)
        if prefill:x=x[-1:]
        return self.dense_linear(self.norm(x,self.w['norm']),self.w['head'])

    @torch.inference_mode()
    def prefill_batch(self,prompts,slots):
        if (not prompts or len(prompts)!=len(slots) or len(set(slots))!=len(slots)
                or any(slot<0 or slot>=self.max_batch for slot in slots)
                or any(not ids or len(ids)>self.context for ids in prompts)):
            raise ValueError('invalid prefill batch')
        return torch.cat([self.prefill(ids,slot) for ids,slot in zip(prompts,slots)])
