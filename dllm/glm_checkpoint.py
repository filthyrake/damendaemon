"""Metadata-only GLM-5.3 main-model checkpoint contract.

Pinned layout: zai-org/GLM-5.3, aca966e4e02791568aa6a4ced368624b3d897f42.
Geometry independently transcribed from Transformers v5.15.0 glm_moe_dsa.
No payload reads, model loading or numerical qualification. MTP is excluded.
"""
import math
from .checkpoint_contract import TensorSpec

DTYPES={'floating':frozenset(('BF16','F32')),
        'fp8':frozenset(('F8_E4M3','F8_E4M3FN')),'scale':frozenset(('F32',))}


def tensor_specs(config):
    if not isinstance(config,dict) or config.get('model_type')!='glm_moe_dsa':
        raise ValueError('expected GLM sparse-attention MoE configuration')
    fields=('hidden_size','vocab_size','num_hidden_layers','num_attention_heads','q_lora_rank',
            'kv_lora_rank','qk_nope_head_dim','qk_rope_head_dim','v_head_dim','index_n_heads',
            'index_head_dim','index_topk','intermediate_size','moe_intermediate_size',
            'n_routed_experts','n_shared_experts','num_experts_per_tok')
    if any(type(config.get(k)) is not int or not 1<=config[k]<=2**21 for k in fields):
        raise ValueError('invalid GLM integer geometry')
    dim,vocab,layers,heads,qr,kr,nope,rope,value,ih,iw,top,dense,moe,experts,shared,selected=(config[k] for k in fields)
    first=config.get('first_k_dense_replace')
    mtp=config.get('num_nextn_predict_layers',0)
    indexers=config.get('indexer_types')
    if (type(first) is not int or not 0<=first<=layers or type(mtp) is not int or not 0<=mtp<=16
            or not isinstance(indexers,list) or len(indexers)!=layers
            or any(kind not in ('full','shared') for kind in indexers) or indexers[0]!='full'
            or config.get('mlp_layer_types')!=['dense']*first+['sparse']*(layers-first)
            or selected>experts or rope%2 or rope>iw
            or type(config.get('qk_head_dim')) is not int or config['qk_head_dim']!=nope+rope
            or type(config.get('num_key_value_heads')) is not int or config['num_key_value_heads']!=heads
            or layers*(6*experts+64)+3>1_000_000):
        raise ValueError('unsupported GLM layer/indexer/expert geometry or inventory size')
    required={'attention_bias':False,'tie_word_embeddings':False,'hidden_act':'silu',
              'rope_interleave':True,'indexer_rope_interleave':True,'scoring_func':'sigmoid',
              'norm_topk_prob':True,'topk_method':'noaux_tc','n_group':1,'topk_group':1}
    if any(config.get(key)!=expected or type(config.get(key)) is not type(expected)
           for key,expected in required.items()):
        raise ValueError('unsupported GLM attention/routing contract')
    position=config.get('rope_parameters')
    if (not isinstance(position,dict) or position.get('rope_type')!='default'
            or type(position.get('rope_theta')) not in (int,float)
            or not 0<position['rope_theta']<float('inf')):
        raise ValueError('unsupported GLM position contract')
    quant=config.get('quantization_config')
    if (not isinstance(quant,dict) or quant.get('quant_method')!='fp8' or quant.get('fmt')!='e4m3'
            or quant.get('activation_scheme')!='dynamic' or quant.get('weight_block_size')!=[128,128]):
        raise ValueError('GLM contract requires blockwise dynamic E4M3 FP8')
    excluded=quant.get('modules_to_not_convert',[])
    if not isinstance(excluded,list) or any(not isinstance(name,str) for name in excluded):
        raise ValueError('invalid GLM quantization exclusions')
    excluded=set(excluded)
    specs={}
    def add(name,shape,storage='floating'):
        if any(size>2**31-1 for size in shape) or math.prod(shape)>2**60:
            raise ValueError('GLM tensor geometry exceeds contract bounds')
        specs[name]=TensorSpec(tuple(shape),storage)
    def linear(name,rows,columns):
        parts=name.split('.')
        if any('.'.join(parts[:size]) in excluded for size in range(1,len(parts)+1)):
            raise ValueError('GLM quantization exclusion changes the pinned matrix layout')
        add(name+'.weight',(rows,columns),'fp8')
        add(name+'.weight_scale_inv',((rows+127)//128,(columns+127)//128),'scale')
    def mlp(name,width):
        linear(name+'.gate_proj',width,dim)
        linear(name+'.up_proj',width,dim)
        linear(name+'.down_proj',dim,width)
    add('model.embed_tokens.weight',(vocab,dim))
    add('lm_head.weight',(vocab,dim))
    add('model.norm.weight',(dim,))
    for layer in range(layers):
        prefix=f'model.layers.{layer}'
        add(prefix+'.input_layernorm.weight',(dim,))
        add(prefix+'.post_attention_layernorm.weight',(dim,))
        attn=prefix+'.self_attn'
        linear(attn+'.q_a_proj',qr,dim)
        add(attn+'.q_a_layernorm.weight',(qr,))
        linear(attn+'.q_b_proj',heads*(nope+rope),qr)
        linear(attn+'.kv_a_proj_with_mqa',kr+rope,dim)
        add(attn+'.kv_a_layernorm.weight',(kr,))
        linear(attn+'.kv_b_proj',heads*(nope+value),kr)
        linear(attn+'.o_proj',dim,heads*value)
        if indexers[layer]=='full':
            index=attn+'.indexer'
            linear(index+'.wq_b',ih*iw,qr)
            linear(index+'.wk',iw,dim)
            add(index+'.k_norm.weight',(iw,))
            add(index+'.k_norm.bias',(iw,))
            add(index+'.weights_proj.weight',(ih,dim))
        if layer<first:
            mlp(prefix+'.mlp',dense)
        else:
            add(prefix+'.mlp.gate.weight',(experts,dim))
            add(prefix+'.mlp.gate.e_score_correction_bias',(experts,))
            mlp(prefix+'.mlp.shared_experts',moe*shared)
            for expert in range(experts):mlp(prefix+f'.mlp.experts.{expert}',moe)
    return specs


def audit_names(config,names):
    specs=tensor_specs(config)
    names=list(names)
    if any(not isinstance(name,str) for name in names) or len(set(names))!=len(names):
        raise ValueError('checkpoint names must be distinct strings')
    layers=config['num_hidden_layers']
    prefixes=tuple(f'model.layers.{i}.' for i in range(layers,layers+config.get('num_nextn_predict_layers',0)))
    excluded=sorted(name for name in names if name.startswith(prefixes))
    main=set(names)-set(excluded)
    missing,extra=sorted(set(specs)-main),sorted(main-set(specs))
    if missing or extra:raise ValueError(f'GLM tensor inventory mismatch: missing={missing[:4]}, extra={extra[:4]}')
    return {'main_tensors':len(specs),'excluded_mtp':excluded}


def audit_headers(config,headers):
    if not isinstance(headers,dict):raise ValueError('checkpoint headers must be a mapping')
    result=audit_names(config,headers)
    for name,spec in tensor_specs(config).items():
        entry=headers[name]
        if (not isinstance(entry,dict) or not isinstance(entry.get('shape'),(list,tuple))
                or any(type(size) is not int for size in entry['shape']) or tuple(entry['shape'])!=spec.shape
                or not isinstance(entry.get('dtype'),str) or entry['dtype'] not in DTYPES[spec.storage]):
            raise ValueError(f'GLM tensor shape/storage mismatch: {name}')
    return result


def audit_snapshot(path,*,names_only=False):
    from .checkpoint_contract import audit_snapshot as audit
    return audit(path,tensor_specs,audit_names,audit_headers,names_only=names_only)
