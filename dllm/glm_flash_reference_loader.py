"""Admitted Flash main-text snapshots into the mixed packed CPU reference stack."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import torch
from .checkpoint import snapshot_shards
from .safetensor_header import read_header,file_stamp,ITEM_BYTES,MAX_HEADER_BYTES,_unique_fields
from .glm_flash_checkpoint import ROOT,tensor_specs,audit_headers
from .packed_checkpoint import _payload
from .packed_moe import PackedMatrix,PackedExpert
from .prepared_projection import PackedProjection
from .glm_flash_linear import FlashLinearAttention
from .glm_flash_indexer import FlashIndexer
from .glm_flash_sparse import FlashSparseAttention
from .glm_flash_hyper import FlashHyperConnection
from .glm_flash_ffn import PackedFlashFeedForward
from .glm_flash_block import FlashBlock
from .glm_flash_model import FlashModel
from .glm_flash_layer_stage import FlashLayerStage,validate_range


def _options(config):
    limits={'num_hidden_layers':128,'num_attention_heads':256,'qk_nope_head_dim':4096,
            'v_head_dim':4096,'index_n_heads':4096,'index_head_dim':4096,'index_kpool':4096,
            'index_topk':4096,'n_routed_experts':4096}
    if any(config[name]>limit for name,limit in limits.items()):
        raise ValueError('Flash geometry exceeds CPU reference component limits')
    if (config['index_topk']%config['index_kpool'] or config['index_topk']<config['index_kpool']
            or config['index_topk']+config['index_kpool']-1>4096):
        raise ValueError('unsupported Flash pool selection width')
    if config['first_k_dense_replace']<config['num_hidden_layers'] and config['n_routed_experts']<2:
        raise ValueError('Flash reference routing requires at least two experts')
    for name,upper in (('rms_norm_eps',1.),('hc_eps',1.),('routed_scaling_factor',torch.finfo(torch.float32).max),
                       ('swiglu_limit',torch.finfo(torch.float32).max)):
        value=config.get(name)
        if type(value) not in (int,float) or not 0<value<=upper:
            raise ValueError(f'invalid Flash reference option: {name}')


def load_reference_model(path,**options):
    """Load the admitted complete main-text model within explicit byte budgets."""
    if 'inspect_only' in options:raise ValueError('use explicit pipeline inspection')
    return _load_reference(path,None,None,**options)


def load_reference_stage(path,start,end,**options):
    """Audit the whole snapshot but read/retain only the requested main-text range."""
    if 'inspect_only' in options:raise ValueError('stage loading requires payloads')
    if start is None or end is None:raise ValueError('Flash stage needs explicit range boundaries')
    return _load_reference(path,start,end,**options)


def _load_reference(path,start,end,*,max_weight_bytes=256*1024**2,projection_workspace_bytes=8*1024**2,
                         max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2,inspect_only=False):
    """Read admitted main-model tensors; leave FP8 weights packed and omit visual/MTP values.

    Weight admission includes retained byte weights/FP32 parameters and one
    largest source tensor for conversion overlap. Header/Python metadata,
    execution outputs/workspace and future KV caches have separate limits.
    At most 256 shard descriptors and 64 MiB aggregate raw headers are admitted.
    This CPU FP32 control does not emulate accelerator activation quantization
    and does not register a tokenizer, GPU backend or HTTP serving model.
    """
    if sys.byteorder!='little':raise ValueError('Flash reference loading requires a little-endian host')
    if (type(max_weight_bytes) is not int or max_weight_bytes<1
            or any(type(v) is not int or not 1<=v<=2**31
                   for v in (projection_workspace_bytes,max_workspace_bytes,max_output_bytes))):
        raise ValueError('invalid Flash reference loader byte budgets')
    path=Path(path)
    with (path/'config.json').open('rb') as reader:raw=reader.read(1024**2+1)
    if len(raw)>1024**2:raise ValueError('Flash configuration exceeds metadata limit')
    config=json.loads(raw,object_pairs_hook=_unique_fields)
    specs=tensor_specs(config);root_config=config;config=root_config['text_config'];_options(config)
    whole=start is None and end is None
    if whole:start,end=0,config['num_hidden_layers']
    validate_range(start,end,config['num_hidden_layers'])
    if not whole:
        prefixes=tuple(ROOT+f'layers.{layer}.' for layer in range(start,end))
        boundary=({ROOT+'embed_tokens.weight'} if start==0 else set())
        if end==config['num_hidden_layers']:boundary|={ROOT+'norm.weight','lm_head.weight'}
        specs={name:spec for name,spec in specs.items() if name in boundary or name.startswith(prefixes)}
    files,index=snapshot_shards(path)
    if len(files)>256:raise ValueError('Flash checkpoint exceeds open-shard limit')
    tensors={}
    with ExitStack() as stack:
        selected={};actual={};opened=[];header_bytes=0
        for file in files:
            reader=stack.enter_context(file.open('rb'))
            entries,base,stamp=read_header(reader,max_header_bytes=min(MAX_HEADER_BYTES,64*1024**2-header_bytes))
            header_bytes+=base-8
            for name,entry in entries.items():
                if name in selected:raise ValueError('duplicate Flash checkpoint tensor')
                selected[name]=(reader,base,entry);actual[name]=str(file.relative_to(path))
            opened.append((reader,stamp))
        if index is not None and actual!=index['weight_map']:
            raise ValueError('Flash checkpoint headers disagree with shard index')
        audit_headers(root_config,{name:entry for name,(_,_,entry) in selected.items()})
        retained=largest=0
        for name,spec in specs.items():
            entry=selected[name][2];count=math.prod(spec.shape)
            retained+=count*(1 if spec.storage=='fp8' else 4)
            largest=max(largest,count*ITEM_BYTES[entry['dtype']])
            if spec.storage=='fp8':
                rows,columns=spec.shape;r,c=min(64,rows),min(256,columns)
                required=128*(r*c+r+c)+8*(r+c)
                if '.mlp.' in name:required+=128*(rows+columns)
                if projection_workspace_bytes<required:
                    raise ValueError('Flash packed projection workspace is insufficient')
        count=config['n_routed_experts'] if config['first_k_dense_replace']<end else 0
        if projection_workspace_bytes+128*(config['hidden_size']+count)>max_workspace_bytes:
            raise ValueError('Flash expert and routing workspace exceeds component allowance')
        if retained+largest>max_weight_bytes:
            raise ValueError('Flash parameters exceed aggregate weight budget')
        if inspect_only:
            if any(file_stamp(reader)!=stamp for reader,stamp in opened):
                raise ValueError('Flash checkpoint changed during header inspection')
            return root_config,retained,largest
        for name,spec in specs.items():
            reader,base,entry=selected[name]
            size=entry['data_offsets'][1]-entry['data_offsets'][0]
            raw=_payload(reader,base,entry|{'shape':[size]})
            dtype=(torch.uint8 if spec.storage=='fp8' else
                   torch.bfloat16 if entry['dtype']=='BF16' else torch.float32)
            value=raw.view(dtype).reshape(spec.shape)
            tensors[name]=value if spec.storage=='fp8' else value.to(torch.float32)
            del raw,value
        if any(file_stamp(reader)!=stamp for reader,stamp in opened):
            raise ValueError('Flash checkpoint changed during payload reads')
    blocks=[]
    def expert(prefix):
        matrices=[PackedMatrix(tensors[prefix+'.'+name+'.weight'],tensors[prefix+'.'+name+'.weight_scale_inv'],'e4m3fn_f32')
                  for name in ('gate_proj','up_proj','down_proj')]
        return PackedExpert(*matrices,workspace_bytes=projection_workspace_bytes,activation_limit=config['swiglu_limit'])
    budgets=dict(max_output_bytes=max_output_bytes,max_workspace_bytes=max_workspace_bytes)
    for layer in range(start,end):
        kind=config['layer_types'][layer]
        prefix=ROOT+f'layers.{layer}';attn=prefix+'.self_attn.'
        weights={}
        for name,spec in specs.items():
            if not name.startswith(attn) or spec.storage=='scale':continue
            value=tensors[name]
            if spec.storage=='fp8':
                value=PackedProjection(value,tensors[name[:-6]+'weight_scale_inv'],format='e4m3fn_f32',
                                       workspace_bytes=projection_workspace_bytes)
            weights[name[len(attn):]]=value
        if kind=='linear_attention':
            lc=config['linear_attn_config']
            attention=FlashLinearAttention(weights,heads=lc['num_heads'],head_dim=lc['head_dim'],
                                            kernel_size=lc['short_conv_kernel_size'],lower_bound=lc['gate_lower_bound'],
                                            epsilon=config['rms_norm_eps'],**budgets)
        else:
            indexed={n[len('indexer.'):]:v for n,v in weights.items() if n.startswith('indexer.')}
            indexer=FlashIndexer(indexed,heads=config['index_n_heads'],pool_size=config['index_kpool'],top_k=config['index_topk'],**budgets)
            attention=FlashSparseAttention({n:v for n,v in weights.items() if not n.startswith('indexer.')},indexer,
                                            heads=config['num_attention_heads'],head_dim=config['qk_nope_head_dim'],
                                            value_dim=config['v_head_dim'],epsilon=config['rms_norm_eps'],**budgets)
        ff_options=dict(max_output_bytes=max_output_bytes,workspace_bytes=max_workspace_bytes,activation_limit=config['swiglu_limit'])
        mlp=prefix+'.mlp'
        if layer<config['first_k_dense_replace']:ffn=PackedFlashFeedForward(expert(mlp),**ff_options)
        else:
            ffn=PackedFlashFeedForward([expert(mlp+f'.experts.{i}') for i in range(config['n_routed_experts'])],
                                    router=tensors[mlp+'.gate.weight'],correction_bias=tensors[mlp+'.gate.e_score_correction_bias'],
                                    shared=expert(mlp+'.shared_experts'),top_k=config['num_experts_per_tok'],
                                    route_scale=config['routed_scaling_factor'],**ff_options)
        def hyper(site):
            return FlashHyperConnection(tensors[prefix+f'.hc_{site}_fn'],tensors[prefix+f'.hc_{site}_base'],
                                        tensors[prefix+f'.hc_{site}_scale'],streams=config['hc_mult'],
                                        norm_epsilon=config['rms_norm_eps'],connection_epsilon=config['hc_eps'],
                                        iterations=config['hc_sinkhorn_iters'],**budgets)
        blocks.append(FlashBlock(attention,ffn,hyper('attn'),hyper('ffn'),tensors[prefix+'.input_layernorm.weight'],
                                 tensors[prefix+'.post_attention_layernorm.weight'],epsilon=config['rms_norm_eps'],**budgets))
    if whole:
        model=FlashModel(tensors[ROOT+'embed_tokens.weight'],blocks,tensors[ROOT+'norm.weight'],tensors['lm_head.weight'],
                         epsilon=config['rms_norm_eps'],max_output_bytes=max_output_bytes)
    else:
        model=FlashLayerStage(blocks,start=start,total=config['num_hidden_layers'],
            embedding=tensors.get(ROOT+'embed_tokens.weight'),norm=tensors.get(ROOT+'norm.weight'),head=tensors.get('lm_head.weight'),
            epsilon=config['rms_norm_eps'],max_output_bytes=max_output_bytes)
    model.config=root_config
    return model
