"""Metadata-only GLM-5.3-Flash main-text checkpoint contract.

Pinned checkpoint eb9eb208eb0d988989d07a6a12d0fdeb5f52574a; independent
shape transcription from Transformers v5.17.0 glm5_next. Raw checkpoint names
precede its conv/projection/hyper-connection renaming. Visual and MTP tensors
are explicitly excluded, not validated or loaded. No weight payloads are read.
"""
import math
from .checkpoint_contract import TensorSpec
from .glm_checkpoint import DTYPES

ROOT='model.language_model.'


def tensor_specs(config):
    if not isinstance(config,dict) or config.get('model_type')!='glm5_next':
        raise ValueError('expected GLM Flash multimodal root configuration')
    text=config.get('text_config');quant=config.get('quantization_config')
    if not isinstance(text,dict) or text.get('model_type')!='glm5_next_text':
        raise ValueError('expected GLM Flash text configuration')
    fields=('hidden_size','vocab_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
            'q_lora_rank','kv_lora_rank','qk_nope_head_dim','qk_head_dim','v_head_dim','index_n_heads',
            'index_head_dim','index_topk','index_kpool','intermediate_size','moe_intermediate_size',
            'n_routed_experts','n_shared_experts','num_experts_per_tok','hc_mult','hc_sinkhorn_iters')
    if any(type(text.get(k)) is not int or not 1<=text[k]<=2**21 for k in fields):
        raise ValueError('invalid GLM Flash text integer geometry')
    dim=text['hidden_size'];layers=text['num_hidden_layers'];experts=text['n_routed_experts']
    first=text.get('first_k_dense_replace');mtp=text.get('num_nextn_predict_layers',0)
    schedule=text.get('layer_types');linear=text.get('linear_attn_config')
    if (type(first) is not int or not 0<=first<=layers or type(mtp) is not int or not 0<=mtp<=16
            or not isinstance(schedule,list) or len(schedule)!=layers
            or any(kind not in ('linear_attention','deepseek_sparse_attention') for kind in schedule)
            or text.get('mlp_layer_types')!=['dense']*first+['sparse']*(layers-first)
            or text.get('indexer_types')!=['full']*layers
            or text['num_experts_per_tok']>experts
            or text['num_attention_heads']!=text['num_key_value_heads']
            or text['qk_head_dim']!=text['qk_nope_head_dim']
            or type(text.get('qk_rope_head_dim')) is not int or text['qk_rope_head_dim']!=0
            or layers*(6*experts+64)+3>1_000_000 or text['hc_mult']>16 or text['hc_sinkhorn_iters']>128):
        raise ValueError('unsupported GLM Flash layer, NoPE, indexer or inventory geometry')
    if (not isinstance(linear,dict)
            or any(type(linear.get(k)) is not int or not 1<=linear[k]<=4096 for k in ('num_heads','head_dim','short_conv_kernel_size'))
            or any(not isinstance(linear.get(k),list) or any(type(i) is not int for i in linear[k])
                   for k in ('kda_layers','full_attn_layers'))
            or linear.get('kda_layers')!=[i for i,k in enumerate(schedule) if k=='linear_attention']
            or linear.get('full_attn_layers')!=[i for i,k in enumerate(schedule) if k=='deepseek_sparse_attention']
            or type(linear.get('gate_lower_bound')) not in (int,float) or not -100<=linear['gate_lower_bound']<0):
        raise ValueError('unsupported GLM Flash KDA geometry or schedule')
    required={'attention_bias':False,'tie_word_embeddings':False,'hidden_act':'silu','scoring_func':'sigmoid',
              'norm_topk_prob':True,'topk_method':'noaux_tc','n_group':1,'topk_group':1,
              'mhc':True,'mla_use_nope':True,'index_kpool_compress':True,'index_kpool_always_select_tail':True}
    if any(text.get(k)!=v or type(text.get(k)) is not type(v) for k,v in required.items()):
        raise ValueError('unsupported GLM Flash attention/routing contract')
    for name in ('rms_norm_eps','hc_eps','swiglu_limit','routed_scaling_factor'):
        value=text.get(name)
        if type(value) not in (int,float) or not 0<value<float('inf'):
            raise ValueError('invalid GLM Flash normalization/routing option')
    if (not isinstance(quant,dict) or quant.get('quant_method')!='fp8' or quant.get('fmt')!='e4m3'
            or quant.get('activation_scheme')!='dynamic' or quant.get('weight_block_size')!=[128,128]):
        raise ValueError('GLM Flash contract requires dynamic block E4M3 FP8')
    exclusions=quant.get('modules_to_not_convert')
    if not isinstance(exclusions,list) or any(not isinstance(v,str) or not v for v in exclusions):
        raise ValueError('invalid GLM Flash quantization exclusions')
    exclusions=set(exclusions)
    def excluded(module):
        # The official exclusion list uses pre-wrapper model.layers names, while
        # the saved main-text namespace is model.language_model.layers.
        for name in (module,module.replace(ROOT,'model.',1)):
            parts=name.split('.')
            if any('.'.join(parts[:i]) in exclusions for i in range(1,len(parts)+1)):
                return True
            if any(part in exclusions for part in parts):return True
        return False
    specs={}
    def add(name,shape,storage='floating'):
        if any(size>2**31-1 for size in shape) or math.prod(shape)>2**60:
            raise ValueError('GLM Flash tensor exceeds shape bounds')
        specs[name]=TensorSpec(tuple(shape),storage)
    def matrix(name,rows,columns,*,packed=True):
        if excluded(name)==packed:raise ValueError('GLM Flash quantization exclusion changes declared storage')
        add(name+'.weight',(rows,columns),'fp8' if packed else 'floating')
        if packed:add(name+'.weight_scale_inv',((rows+127)//128,(columns+127)//128),'scale')
    def mlp(prefix,width):
        matrix(prefix+'.gate_proj',width,dim);matrix(prefix+'.up_proj',width,dim);matrix(prefix+'.down_proj',dim,width)
    add(ROOT+'embed_tokens.weight',(text['vocab_size'],dim));add('lm_head.weight',(text['vocab_size'],dim))
    add(ROOT+'norm.weight',(dim,))
    heads=text['num_attention_heads'];qr=text['q_lora_rank'];kr=text['kv_lora_rank'];nope=text['qk_nope_head_dim']
    ih=text['index_n_heads'];iw=text['index_head_dim'];hc=text['hc_mult'];mix=hc*(2+hc)
    lh=linear['num_heads'];lw=linear['head_dim'];lq=lh*lw
    for layer,kind in enumerate(schedule):
        prefix=ROOT+f'layers.{layer}';attn=prefix+'.self_attn'
        add(prefix+'.input_layernorm.weight',(dim,));add(prefix+'.post_attention_layernorm.weight',(dim,))
        for sublayer in ('attn','ffn'):
            add(prefix+f'.hc_{sublayer}_fn',(mix,hc*dim));add(prefix+f'.hc_{sublayer}_base',(mix,));add(prefix+f'.hc_{sublayer}_scale',(3,))
        if kind=='linear_attention':
            for name in ('q','k','v'):
                matrix(attn+f'.{name}_proj',lq,dim,packed=False)
                add(attn+f'.{name}_conv1d.weight',(lq,1,linear['short_conv_kernel_size']))
            for name in ('f','g'):
                matrix(attn+f'.{name}_a_proj',lw,dim,packed=False);matrix(attn+f'.{name}_b_proj',lq,lw,packed=False)
            matrix(attn+'.b_proj',lh,dim,packed=False);matrix(attn+'.o_proj',dim,lq,packed=False)
            add(attn+'.o_norm.weight',(lw,));add(attn+'.A_log',(lh,));add(attn+'.dt_bias',(lq,))
        else:
            matrix(attn+'.q_a_proj',qr,dim);add(attn+'.q_a_layernorm.weight',(qr,))
            matrix(attn+'.q_b_proj',heads*nope,qr)
            matrix(attn+'.kv_a_proj_with_mqa',kr,dim);add(attn+'.kv_a_layernorm.weight',(kr,))
            matrix(attn+'.kv_b_proj',heads*(nope+text['v_head_dim']),kr,packed=False)
            matrix(attn+'.o_proj',dim,heads*text['v_head_dim'])
            index=attn+'.indexer'
            matrix(index+'.wq_b',ih*iw,qr,packed=False);matrix(index+'.wk',iw,dim,packed=False)
            matrix(index+'.weights_proj',ih,dim,packed=False)
            add(index+'.k_norm.weight',(iw,));add(index+'.k_norm.bias',(iw,))
            add(index+'.index_kpool_compress_ape',(text['index_kpool'],iw))
            add(index+'.index_kpool_compress_gate',(iw,dim))
        if layer<first:mlp(prefix+'.mlp',text['intermediate_size'])
        else:
            add(prefix+'.mlp.gate.weight',(experts,dim));add(prefix+'.mlp.gate.e_score_correction_bias',(experts,))
            mlp(prefix+'.mlp.shared_experts',text['moe_intermediate_size']*text['n_shared_experts'])
            for expert in range(experts):mlp(prefix+f'.mlp.experts.{expert}',text['moe_intermediate_size'])
    return specs


def audit_names(config,names):
    specs=tensor_specs(config);names=list(names)
    if any(not isinstance(n,str) for n in names) or len(names)!=len(set(names)):
        raise ValueError('GLM Flash tensor names must be distinct strings')
    text=config['text_config'];layers=text['num_hidden_layers']
    prefixes=tuple(ROOT+f'layers.{i}.' for i in range(layers,layers+text.get('num_nextn_predict_layers',0)))
    mtp=sorted(n for n in names if n.startswith(prefixes))
    vision=sorted(n for n in names if n.startswith('model.visual.'))
    main=set(names)-set(mtp)-set(vision);missing=set(specs)-main;extra=main-set(specs)
    if missing or extra:raise ValueError(f'GLM Flash text inventory mismatch: missing={sorted(missing)[:4]}, extra={sorted(extra)[:4]}')
    return dict(main_tensors=len(specs),excluded_mtp=mtp,excluded_vision=vision)


def audit_headers(config,headers):
    if not isinstance(headers,dict):raise ValueError('GLM Flash headers must be a mapping')
    report=audit_names(config,headers)
    for name,spec in tensor_specs(config).items():
        entry=headers[name]
        if (not isinstance(entry,dict) or not isinstance(entry.get('shape'),(list,tuple))
                or any(type(v) is not int for v in entry['shape']) or tuple(entry['shape'])!=spec.shape
                or not isinstance(entry.get('dtype'),str) or entry['dtype'] not in DTYPES[spec.storage]):
            raise ValueError(f'GLM Flash tensor shape/storage mismatch: {name}')
    return report


def audit_snapshot(path,*,names_only=False):
    from .checkpoint_contract import audit_snapshot as audit
    return audit(path,tensor_specs,audit_names,audit_headers,names_only=names_only)
