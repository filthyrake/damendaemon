"""Header-only declared-storage placement for explicit modern CPU host stages."""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from .checkpoint import snapshot_shards
from .reference_pipeline_build import snapshot_inventory
from .safetensor_header import read_header,file_stamp,ITEM_BYTES,MAX_HEADER_BYTES


@dataclass(frozen=True)
class ReferenceStorageLayout:
    family: str
    config_json: str
    path: Path
    receipt: tuple
    # First boundary, each global layer, final boundary: retained/source maximum.
    components: tuple
    metadata_id: str
    inspection_limits: tuple

    @property
    def config(self):return json.loads(self.config_json)

    def verify_unchanged(self):
        if snapshot_inventory(self.path)!=self.receipt:raise ValueError('reference storage snapshot changed during planning')


def inspect_reference_storage(path,family,*,projection_workspace_bytes=8*1024**2,
                              max_workspace_bytes=64*1024**2,max_output_bytes=64*1024**2):
    """Audit via existing family loaders, then read physical header sizes once.

    Two bounded header passes independent of candidate count; no payload reads.
    Receipts detect ordinary concurrent file changes, not authenticate model bytes.
    """
    if family not in ('glm','flash','deepseek'):raise ValueError('unsupported CPU reference storage family')
    if any(type(v) is not int or not 1<=v<=2**31 for v in (projection_workspace_bytes,max_workspace_bytes,max_output_bytes)):
        raise ValueError('invalid CPU storage inspection workspace/output bounds')
    path=Path(path).expanduser().resolve();receipt=snapshot_inventory(path)
    options=dict(max_weight_bytes=2**40,max_output_bytes=max_output_bytes)
    if family=='deepseek':
        from .deepseek_checkpoint_pipeline import _inspect
        from .deepseek_checkpoint import tensor_specs
        options['workspace_bytes']=projection_workspace_bytes
        prefix='layers.';embedding='embed.weight'
    else:
        options.update(projection_workspace_bytes=projection_workspace_bytes,max_workspace_bytes=max_workspace_bytes)
        if family=='glm':
            from .glm_checkpoint_pipeline import _inspect
            from .glm_checkpoint import tensor_specs
            prefix='model.layers.';embedding='model.embed_tokens.weight'
        else:
            from .glm_flash_checkpoint_pipeline import _inspect
            from .glm_flash_checkpoint import tensor_specs,ROOT
            prefix=ROOT+'layers.';embedding=ROOT+'embed_tokens.weight'
    config,retained,largest=_inspect(path,**options)
    text=config['text_config'] if family=='flash' else config;count=text['num_hidden_layers']
    if not 2<=count<=128:raise ValueError('CPU host storage planning needs 2..128 layers')
    specs=tensor_specs(config);components=[[0,0] for _ in range(count+2)];headers={};header_bytes=0
    files,_=snapshot_shards(path)
    for file in files:
        with file.open('rb') as reader:
            entries,base,stamp=read_header(reader,max_header_bytes=min(MAX_HEADER_BYTES,64*1024**2-header_bytes))
            header_bytes+=base-8
            if set(headers)&set(entries):raise ValueError('duplicate tensor during storage inspection')
            headers.update(entries)
            if file_stamp(reader)!=stamp:raise ValueError('reference storage header changed during inspection')
    inventory=[]
    for name,spec in specs.items():
        entry=headers[name];elements=math.prod(spec.shape);source=elements*ITEM_BYTES[entry['dtype']]
        item_bytes=(4 if spec.storage=='floating' else ITEM_BYTES[entry['dtype']] if spec.storage=='integer' else 1) if family=='deepseek' else (1 if spec.storage=='fp8' else 4)
        if name==embedding:index=0
        elif name.startswith(prefix):index=1+int(name[len(prefix):].split('.',1)[0])
        else:index=count+1
        components[index][0]+=elements*item_bytes;components[index][1]=max(components[index][1],source)
        inventory.append((name,spec.shape,entry['dtype'],item_bytes))
    if sum(v[0] for v in components)!=retained or max(v[1] for v in components)!=largest:
        raise ValueError('reference storage header accounting disagrees with loader admission')
    config_json=json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=hashlib.sha256(json.dumps([config,inventory],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result=ReferenceStorageLayout(family,config_json,path,receipt,tuple(map(tuple,components)),digest,
        (projection_workspace_bytes,max_workspace_bytes,max_output_bytes))
    result.verify_unchanged();return result


def _cache_per_layer(family,c,context):
    if family=='glm':
        return tuple(4*context*(c['kv_lora_rank']+c['qk_rope_head_dim']+(c['index_head_dim'] if kind=='full' else 0)) for kind in c['indexer_types'])
    if family=='flash':
        linear=c['linear_attn_config'];heads,width,kernel=(linear[k] for k in ('num_heads','head_dim','short_conv_kernel_size'))
        recurrent=4*(3*heads*width*(kernel-1)+heads*width*width)
        sparse=4*context*(c['kv_lora_rank']+2*c['index_head_dim'])
        return tuple(recurrent if kind=='linear_attention' else sparse for kind in c['layer_types'])
    from .floating_attention import cache_bytes_per_slot
    return tuple(cache_bytes_per_slot(c['sliding_window'],ratio,context,c['head_dim'],c['index_head_dim'],4)
                 for ratio in c['compress_ratios'][:c['num_hidden_layers']])


def plan_reference_stages(layout,budgets,*,context=512,slots=4,chunk_size=128,
        max_weight_bytes=256*1024**2,max_cache_bytes=256*1024**2,
        projection_workspace_bytes=8*1024**2,max_workspace_bytes=64*1024**2,
        max_output_bytes=64*1024**2,frame_bytes=64*1024**2):
    """Minimize maximum declared tensor/allowance utilization; never admit execution.

    Caller budgets reserve unrelated memory. Python, allocator and OS overhead,
    kernels and unbounded arithmetic temporaries are not measured by this report.
    DeepSeek's extra workspace amount is an allowance, not an enforced runtime cap.
    """
    if not isinstance(layout,ReferenceStorageLayout):raise ValueError('storage planning requires an inspected snapshot')
    root=layout.config;c=root['text_config'] if layout.family=='flash' else root;count=c['num_hidden_layers']
    limit=c.get('max_position_embeddings',2**20)
    if type(limit) is not int or not 1<=limit<=2**31:raise ValueError('invalid model position limit in storage metadata')
    if (not isinstance(budgets,(list,tuple)) or not 2<=len(budgets)<=min(8,count)
            or any(type(v) is not int or not 1<=v<=2**40 for v in budgets)
            or type(context) is not int or not 1<=context<=min(2**20,limit)
            or type(slots) is not int or not 1<=slots<=4 or type(chunk_size) is not int or not 1<=chunk_size<=4096
            or any(type(v) is not int or not 1<=v<=2**31 for v in (max_weight_bytes,max_cache_bytes,projection_workspace_bytes,max_workspace_bytes,max_output_bytes))
            or type(frame_bytes) is not int or not 256<=frame_bytes<=64*1024**2):
        raise ValueError('invalid reference storage placement controls')
    layout.verify_unchanged()
    # Re-audit workspace options without payloads if the caller changes them.
    if (projection_workspace_bytes,max_workspace_bytes,max_output_bytes)!=layout.inspection_limits:
        checked=inspect_reference_storage(layout.path,layout.family,projection_workspace_bytes=projection_workspace_bytes,
            max_workspace_bytes=max_workspace_bytes,max_output_bytes=max_output_bytes)
        if checked.receipt!=layout.receipt or checked.metadata_id!=layout.metadata_id:raise ValueError('reference storage snapshot changed before placement')
    width=c['hidden_size']*(c['hc_mult'] if layout.family in ('flash','deepseek') else 1)
    rows=max(slots,min(context,chunk_size));activation=rows*width*4
    token_frame=rows*8 if layout.family=='deepseek' else 0
    caches=_cache_per_layer(layout.family,c,context)
    cuts={0,count}|({i for i,kind in enumerate(c['indexer_types']) if kind=='full'} if layout.family=='glm' else set(range(count)))
    if layout.family=='glm':
        from .glm_layer_stage import validate_range
        validate_range(0,count,count,c['indexer_types'])
    candidates={}
    for start in sorted(cuts):
        retained=0;overlap=0;cache=0
        if start==0:retained,overlap=layout.components[0]
        for end in range(start+1,count+1):
            weight,source=layout.components[end];retained+=weight;overlap=max(overlap,source);cache+=slots*caches[end-1]
            if end not in cuts:continue
            final_weight,final_source=layout.components[-1] if end==count else (0,0)
            stage_weight=retained+final_weight;conversion=max(overlap,final_source)
            output=max(activation,slots*c['vocab_size']*4 if end==count else 0)
            if (stage_weight+conversion>max_weight_bytes or cache>max_cache_bytes or output>max_output_bytes
                    or max(activation,token_frame)>frame_bytes):continue
            allowances=projection_workspace_bytes+max_workspace_bytes+max_output_bytes+2*activation+token_frame
            candidates[(start,end)]=dict(layers=[start,end],retained_weight_bytes=stage_weight,
                source_conversion_overlap_bytes=conversion,cache_reservation_bytes=cache,
                projection_workspace_bytes=projection_workspace_bytes,component_workspace_allowance_bytes=max_workspace_bytes,
                output_allowance_bytes=max_output_bytes,transport_buffer_allowance_bytes=2*activation+token_frame,
                required_storage_and_allowance_bytes=stage_weight+conversion+cache+allowances)
    states={0:(Fraction(0),())}
    for rank,limit in enumerate(budgets):
        following={}
        for start,(score,ranges) in states.items():
            for end in range(start+1,count-(len(budgets)-rank-1)+1):
                candidate=candidates.get((start,end))
                if candidate is None or candidate['required_storage_and_allowance_bytes']>limit:continue
                choice=(max(score,Fraction(candidate['required_storage_and_allowance_bytes'],limit)),(*ranges,(start,end)))
                if end not in following or choice<following[end]:following[end]=choice
        states=following
    report=dict(scope='cpu_reference_declared_storage',family=layout.family,context=context,slots=slots,chunk_size=chunk_size,
        cpu_budget_bytes=list(budgets),metadata_id=layout.metadata_id,source_model_identity_verified=False,
        execution_admitted=False,official_model_qualified=False,storage_plan_found=count in states,
        excluded_memory=['Python objects','allocator and OS overhead','derived preparation tensors','unmeasured arithmetic temporaries'],
        component_workspace_cap_enforced=layout.family!='deepseek')
    if count in states:
        score,ranges=states[count]
        report.update(plan={'ranges':[list(pair) for pair in ranges]},maximum_declared_utilization=float(score),
            stages=[dict(candidates[pair],cpu_budget_bytes=limit,remaining_budget_bytes=limit-candidates[pair]['required_storage_and_allowance_bytes'])
                    for pair,limit in zip(ranges,budgets)])
    else:report['reason']='No complete contiguous partition fits declared budgets and unchanged stage/context/slot limits'
    layout.verify_unchanged();return report
