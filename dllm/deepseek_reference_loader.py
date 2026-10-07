"""Explicit budgeted CPU reference snapshot loading; no serving admission."""
from contextlib import ExitStack
import json
import math
from pathlib import Path

import torch

from .checkpoint import snapshot_shards, _unique_fields
from .deepseek_checkpoint import tensor_specs, audit_headers
from .deepseek_position import frequencies
from .floating_block import FloatingBlock
from .floating_model import FloatingModel
from .packed_checkpoint import _payload
from .packed_moe import PackedMatrix, PackedExpert
from .prepared_projection import PackedProjection
from .safetensor_header import read_header, file_stamp, ITEM_BYTES, MAX_HEADER_BYTES

DTYPES={'BF16':torch.bfloat16,'F32':torch.float32,'I32':torch.int32,'I64':torch.int64}


def _options(config):
    if (config.get('scoring_func') != 'sqrtsoftplus' or config.get('norm_topk_prob') is not True
            or config.get('topk_method') != 'noaux_tc' or config.get('hidden_act') != 'silu'
            or config.get('tie_word_embeddings') is not False):
        raise ValueError('unsupported reference model routing/activation/embedding configuration')
    ints={'sliding_window':(1,2**31-1),'hc_sinkhorn_iters':(1,128),'index_topk':(1,4096),
          'qk_rope_head_dim':(2,4096)}
    for name,(lo,hi) in ints.items():
        value=config.get(name)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f'invalid reference model option: {name}')
    for name in ('rms_norm_eps','hc_eps','swiglu_limit','routed_scaling_factor'):
        value=config.get(name)
        if type(value) not in (int,float) or not 0 < value <= torch.finfo(torch.float32).max:
            raise ValueError(f'invalid reference model option: {name}')
    rope=config.get('rope_scaling')
    if not isinstance(rope,dict) or rope.get('type') != 'yarn':
        raise ValueError('reference loader requires explicit YaRN configuration')
    rotary=config['qk_rope_head_dim']
    if (config['hc_mult'] > 16 or rotary > min(config['head_dim'],config['index_head_dim'])
            or (config['head_dim']-rotary)%64 or not 32 <= config['index_head_dim'] <= 4096
            or config['index_head_dim'] & (config['index_head_dim']-1)):
        raise ValueError('unsupported reference model rotary/hyper-connection geometry')
    result={}
    for ratio in set(config['compress_ratios'][:config['num_hidden_layers']]):
        result[ratio]=frequencies(rotary,base=config.get('compress_rope_theta') if ratio else config.get('rope_theta'),
                                 original_length=rope.get('original_max_position_embeddings') if ratio else 0,
                                 factor=rope.get('factor'),beta_fast=rope.get('beta_fast'),beta_slow=rope.get('beta_slow'))
    return result


def load_reference_model(path, *, max_weight_bytes=256*1024*1024, dtype=torch.float32,
                         workspace_bytes=8*1024*1024):
    """Load main-model parameters into the explicit packed CPU reference stack.

    Admit retained raw/converted parameter bytes plus one largest source tensor
    for conversion overlap before any payload read. Header/Python metadata,
    projection outputs/workspace and future KV caches are separate. At most 256
    shard descriptors and 64 MiB aggregate raw headers are admitted. FP32 is the
    default reference storage; BF16/FP16 must be chosen explicitly. MTP is omitted.
    No tokenizer, template, GPU backend or HTTP model support is registered.
    """
    if (type(max_weight_bytes) is not int or max_weight_bytes < 1
            or type(workspace_bytes) is not int or workspace_bytes < 1
            or dtype not in (torch.float32,torch.bfloat16,torch.float16)):
        raise ValueError('invalid reference loader byte budget or compute dtype')
    path=Path(path)
    config_path=path/'config.json'
    if config_path.stat().st_size > 1024*1024:
        raise ValueError('reference model configuration exceeds size limit')
    config=json.loads(config_path.read_text(),object_pairs_hook=_unique_fields)
    specs=tensor_specs(config)
    freqs=_options(config)
    files,index=snapshot_shards(path)
    if len(files) > 256:
        raise ValueError('reference checkpoint exceeds open-shard limit')
    tensors={}
    with ExitStack() as stack:
        selected={}
        actual={}
        opened=[]
        header_bytes=0
        for file in files:
            reader=stack.enter_context(file.open('rb'))
            entries,base,stamp=read_header(reader,max_header_bytes=min(MAX_HEADER_BYTES,64*1024*1024-header_bytes))
            header_bytes+=base-8
            if header_bytes > 64*1024*1024:
                raise ValueError('reference checkpoint exceeds aggregate header limit')
            for name,entry in entries.items():
                if name in selected:
                    raise ValueError('duplicate checkpoint tensor during reference loading')
                selected[name]=(reader,base,entry)
                actual[name]=str(file.relative_to(path))
            opened.append((reader,stamp))
        if index is not None and actual != index['weight_map']:
            raise ValueError('checkpoint headers disagree with shard index')
        audit_headers(config,{name:entry for name,(_,_,entry) in selected.items()})
        targets={}
        retained=largest=0
        for name,spec in specs.items():
            entry=selected[name][2]
            source_bytes=math.prod(spec.shape)*ITEM_BYTES[entry['dtype']]
            if spec.storage in ('fp4','fp8'):
                rows,columns=spec.shape
                if spec.storage == 'fp4':columns*=2
                r,c=min(64,rows),min(256,columns)
                required=128*(r*c+r+c)+8*(r+c)
                if '.ffn.' in name:
                    required+=128*(config['hidden_size']+config['moe_intermediate_size'])
                if workspace_bytes < required:
                    raise ValueError('reference model projection workspace is insufficient')
            if spec.storage == 'floating':
                target=dtype if name == 'embed.weight' or name.endswith('.indexer.weights_proj.weight') else torch.float32
            elif spec.storage == 'integer':target=DTYPES[entry['dtype']]
            else:target=torch.uint8
            targets[name]=target
            retained+=math.prod(spec.shape)*torch.empty((),dtype=target).element_size()
            largest=max(largest,source_bytes)
        if retained+largest > max_weight_bytes:
            raise ValueError('reference model parameter bytes exceed aggregate weight budget')
        for name,spec in specs.items():
            reader,base,entry=selected[name]
            count=entry['data_offsets'][1]-entry['data_offsets'][0]
            raw=_payload(reader,base,entry|{'shape':[count]})
            source=DTYPES[entry['dtype']] if spec.storage in ('floating','integer') else torch.uint8
            tensors[name]=raw.view(source).reshape(spec.shape).to(targets[name])
            del raw
        if any(file_stamp(reader) != stamp for reader,stamp in opened):
            raise ValueError('checkpoint changed during reference model loading')
    blocks=[]
    def matrix(name,format):
        return PackedMatrix(tensors[name+'.weight'],tensors[name+'.scale'],format)
    for layer,ratio in enumerate(config['compress_ratios'][:config['num_hidden_layers']]):
        prefix=f'layers.{layer}.'
        attention={}
        weights={}
        for name,spec in specs.items():
            if not name.startswith(prefix):continue
            local=name[len(prefix):]
            if local.startswith('attn.'):
                if spec.storage == 'scale':continue
                if spec.storage == 'fp8':
                    value=PackedProjection(tensors[name],tensors[name[:-6]+'scale'],format='e4m3fn',dtype=dtype,
                                           workspace_bytes=workspace_bytes)
                else:value=tensors[name]
                attention[local[5:]]=value
            elif not local.startswith(('ffn.experts.','ffn.shared_experts.')):
                weights[local]=tensors[name]
        experts=[]
        for expert in range(config['n_routed_experts']+1):
            shared=expert == config['n_routed_experts']
            name=prefix+'ffn.'+('shared_experts' if shared else f'experts.{expert}')
            format='e4m3fn' if shared else 'e2m1'
            experts.append(PackedExpert(*(matrix(name+'.'+projection,format) for projection in ('w1','w3','w2')),
                                        activation_limit=config['swiglu_limit'],workspace_bytes=workspace_bytes))
        blocks.append(FloatingBlock(attention,weights,freqs[ratio],groups=config['o_groups'],window=config['sliding_window'],
                                    ratio=ratio,hc_mult=config['hc_mult'],top_k=config['num_experts_per_tok'],
                                    route_scale=config['routed_scaling_factor'],index_top_k=config['index_topk'],
                                    epsilon=config['rms_norm_eps'],connection_epsilon=config['hc_eps'],
                                    iterations=config['hc_sinkhorn_iters'],activation_limit=config['swiglu_limit'],
                                    workspace_bytes=workspace_bytes,packed_experts=experts[:-1],packed_shared=experts[-1]))
    head={name:tensors[name] for name in ('head.weight','norm.weight','hc_head_fn','hc_head_scale','hc_head_base')}
    model=FloatingModel(tensors['embed.weight'],blocks,head,epsilon=config['rms_norm_eps'],connection_epsilon=config['hc_eps'])
    model.config=dict(config)
    return model
