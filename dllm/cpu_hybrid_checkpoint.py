"""Header-admitted text-only Qwen3.5 snapshots for explicit CPU pipelines.

No accelerator selection, tokenizer configuration or serving registration occurs.
Source floating weights are converted one fused group at a time; all packed
matrices retain canonical Q4_0/Q8_0 storage and FP16 scales.
"""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from .checkpoint import snapshot_shards
from .config import hybrid_config
from .cpu_hybrid_stage import cpu_hybrid_shapes
from .packed_checkpoint import _payload
from .quantization import quantize,quantize_q4
from .safetensor_header import read_header,file_stamp,MAX_HEADER_BYTES,_unique_fields


def _groups(c,prefix):
    """Internal name -> ordered (source name, shape) pieces and norm offset."""
    shapes=cpu_hybrid_shapes(c,0,c['num_hidden_layers']);groups={}
    def add(name,sources,offset=False):groups[name]=(sources,offset)
    add('embed',[(prefix+'embed_tokens.weight',shapes['embed'])])
    add('norm',[(prefix+'norm.weight',shapes['norm'])],True)
    if not c['tie_word_embeddings']:add('head',[('lm_head.weight',shapes['head'])])
    d=c['hidden_size'];inter=c['intermediate_size'];hd=c['head_dim']
    for i,kind in enumerate(c['layer_types']):
        root=prefix+f'layers.{i}.'
        for short,long in (('norm1','input_layernorm'),('norm2','post_attention_layernorm')):
            add(f'{i}.{short}',[(root+long+'.weight',(d,))],True)
        add(f'{i}.gate_up',[(root+'mlp.'+n+'_proj.weight',(inter,d)) for n in ('gate','up')])
        add(f'{i}.down',[(root+'mlp.down_proj.weight',shapes[f'{i}.down'])])
        if kind=='full_attention':
            h=c['num_attention_heads'];kh=c['num_key_value_heads']
            add(f'{i}.qkv',[(root+'self_attn.'+n+'_proj.weight',(rows,d))
                           for n,rows in (('q',2*h*hd),('k',kh*hd),('v',kh*hd))])
            for short,long in (('o','o_proj'),('q_norm','q_norm'),('k_norm','k_norm')):
                add(f'{i}.{short}',[(root+'self_attn.'+long+'.weight',shapes[f'{i}.{short}'])],short.endswith('norm'))
        else:
            for short,long in (('qkv','in_proj_qkv.weight'),('z','in_proj_z.weight'),('b','in_proj_b.weight'),
                               ('a','in_proj_a.weight'),('out','out_proj.weight'),('conv','conv1d.weight'),
                               ('dt','dt_bias'),('A','A_log'),('gated_norm','norm.weight')):
                add(f'{i}.{short}',[(root+'linear_attn.'+long,shapes[f'{i}.{short}'])])
    return shapes,groups


def load_cpu_checkpoint(path,*,quantization='none',max_context=512,max_weight_bytes=256*1024**2,
                        max_workspace_bytes=64*1024**2):
    """Load bounded floating safetensors into a prepared CPU model.

    max_weight_bytes covers retained weights/scales. max_workspace_bytes covers
    a conservative peak source/cast/fusion/quantization overlap; both are admitted
    before payload reads. Metadata/allocator overhead is separate. Text-only
    configurations and exact inventories are required; visual/MTP extras reject.
    """
    if (sys.byteorder!='little' or quantization not in ('none','q4_0','q8_0')
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_weight_bytes,max_workspace_bytes))):
        raise ValueError('invalid CPU checkpoint format/budgets')
    path=Path(path)
    with (path/'config.json').open('rb') as reader:raw=reader.read(1024**2+1)
    if len(raw)>1024**2:raise ValueError('CPU checkpoint configuration exceeds metadata limit')
    source=json.loads(raw,object_pairs_hook=_unique_fields)
    if not isinstance(source,dict) or source.get('model_type')!='qwen3_5_text' or 'text_config' in source or 'vision_config' in source:
        raise ValueError('CPU checkpoint requires a text-only Qwen3.5 configuration')
    try:c=hybrid_config(source)
    except (TypeError,OverflowError,AttributeError) as error:raise ValueError('invalid CPU checkpoint configuration') from error
    if type(c.get('tie_word_embeddings')) is not bool:raise ValueError('invalid CPU checkpoint tied embedding flag')
    shapes,_=_groups(c,'model.')
    if type(max_context) is not int or not 1<=max_context<=min(2**20,c['max_position_embeddings']):
        raise ValueError('invalid CPU checkpoint context')
    packed={n for n,shape in shapes.items() if quantization!='none' and n!='embed' and len(shape)==2}
    if any(shapes[n][1]%32 for n in packed):raise ValueError('CPU checkpoint packed matrix width must divide by 32')
    retained=sum(math.prod(shape)*4 if n not in packed else
                 math.prod(shape)//(2 if quantization=='q4_0' else 1)+math.prod(shape)//32*2
                 for n,shape in shapes.items())
    if c['tie_word_embeddings'] and quantization=='none':retained-=math.prod(shapes['head'])*4
    # 32 bytes per largest fused element conservatively bounds source bytes,
    # FP32 pieces + concatenation and the quantizer's 128-row temporaries.
    workspace=32*max(math.prod(s) for s in shapes.values())
    if retained>max_weight_bytes or workspace>max_workspace_bytes:raise ValueError('CPU checkpoint weight/workspace budget exceeded')
    files,index=snapshot_shards(path)
    if len(files)>256:raise ValueError('CPU checkpoint descriptor limit exceeded')
    weights={};scales={}
    with ExitStack() as stack:
        entries={};actual={};opened=[];header_bytes=0
        for file in files:
            reader=stack.enter_context(file.open('rb'))
            header,base,stamp=read_header(reader,max_header_bytes=min(MAX_HEADER_BYTES,64*1024**2-header_bytes))
            header_bytes+=base-8;opened.append((reader,stamp))
            for name,entry in header.items():
                if name in entries:raise ValueError('duplicate CPU checkpoint tensor')
                entries[name]=(reader,base,entry);actual[name]=str(file.relative_to(path))
        if index is not None and actual!=index['weight_map']:raise ValueError('CPU checkpoint headers disagree with shard index')
        prefix='model.language_model.' if 'model.language_model.embed_tokens.weight' in entries else 'model.'
        _,groups=_groups(c,prefix)
        expected={name:shape for pieces,_ in groups.values() for name,shape in pieces}
        if set(entries)!=set(expected):raise ValueError('CPU checkpoint tensor inventory mismatch')
        for name,shape in expected.items():
            entry=entries[name][2]
            if entry['dtype'] not in ('F16','BF16','F32') or tuple(entry['shape'])!=shape:
                raise ValueError('CPU checkpoint tensor shape/dtype mismatch')
        for name,(pieces,offset) in groups.items():
            values=[]
            for source_name,shape in pieces:
                reader,base,entry=entries[source_name];size=entry['data_offsets'][1]-entry['data_offsets'][0]
                raw=_payload(reader,base,entry|{'shape':[size]})
                value=raw.view({'F16':torch.float16,'BF16':torch.bfloat16,'F32':torch.float32}[entry['dtype']]).reshape(shape).float()
                if not torch.isfinite(value).all():raise ValueError('nonfinite CPU checkpoint parameter')
                values.append(value);del raw,value
            value=values[0] if len(values)==1 else torch.cat(values,dim=0)
            del values
            if offset:value.add_(1.)
            if name in packed:weights[name],scales[name]=(quantize_q4 if quantization=='q4_0' else quantize)(value)
            else:weights[name]=value
            del value
        if c['tie_word_embeddings']:
            if quantization=='none':weights['head']=weights['embed']
            else:weights['head'],scales['head']=(quantize_q4 if quantization=='q4_0' else quantize)(weights['embed'])
        if any(file_stamp(reader)!=stamp for reader,stamp in opened):raise ValueError('CPU checkpoint changed during reads')
    return SimpleNamespace(config=c,source_config=source,weights=weights,scales=scales,quantization=quantization,
                           max_context=max_context,vision_weights=None,bytes=int(retained),
                           loader_workspace_bytes=workspace)
