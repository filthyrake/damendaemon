"""Explicit prepared DeepSeek placement with complete pre-copy metadata admission."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from .floating_model import FloatingModel
from .floating_block import FloatingBlock
from .prepared_projection import PackedProjection
from .packed_moe import PackedExpert,PackedMatrix
from .cuda_deepseek_model import CudaDeepSeekModel
from .cuda_deepseek_block import CudaDeepSeekBlock
from .cuda_prepared_projection import CudaPackedProjection
from .cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
from .cuda_packed_placement import _copy


@dataclass(frozen=True)
class PlacedCudaDeepSeekModel:
    model: CudaDeepSeekModel
    raw_bytes: int
    device: torch.device


def _tensors(source):
    if not isinstance(source,FloatingModel) or isinstance(source,CudaDeepSeekModel):
        raise ValueError('placement requires a prepared CPU DeepSeek model')
    tensors={}
    def add(value):
        if isinstance(value,PackedProjection):add(value.weight);add(value.scales)
        elif isinstance(value,torch.Tensor) and value.device.type=='cpu':tensors[id(value)]=value
        else:raise ValueError('DeepSeek placement requires CPU tensors or packed projections')
    add(source.embedding)
    for value in source.head.values():add(value)
    for block in source.blocks:
        if (not isinstance(block,FloatingBlock) or block.backend is not None
                or not isinstance(block.packed_experts,(list,tuple)) or not block.packed_experts):
            raise ValueError('DeepSeek placement requires CPU packed blocks')
        for value in (*block.weights.values(),*block.attention_weights.values(),block.frequencies):add(value)
        for expert in (*block.packed_experts,block.packed_shared):
            if not isinstance(expert,PackedExpert):raise ValueError('DeepSeek placement requires CPU packed experts')
            for matrix in (expert.gate,expert.up,expert.down):add(matrix.weight);add(matrix.scales)
    return tensors


def _inner_workspace(block):
    experts=(*block.packed_experts,block.packed_shared)
    table=block.weights.get('ffn.gate.tid2eid')
    validation=8*block.weights['ffn.gate.weight'].numel()+(0 if table is None else 32*table.numel())
    scratch=max(e.workspace_bytes for e in experts)+128*(block.dim+len(block.packed_experts)+block.top_k)
    return max(block.workspace_bytes,validation,scratch)


def _policy(source,workspace_bytes,max_output_bytes):
    if (type(workspace_bytes) is not int or not 4<=workspace_bytes<=2**31
            or type(max_output_bytes) is not int or not 1<=max_output_bytes<=2**31
            or not 1<=len(source.blocks)<=1024 or any(not 1<=n<=2**21 for n in source.embedding.shape)):
        raise ValueError('invalid DeepSeek CUDA model geometry/budgets')
    values=[source.embedding,*source.head.values()]
    for block in source.blocks:
        values.extend(v for k,v in block.weights.items() if k!='ffn.gate.tid2eid')
        values.extend(block.attention_weights.values());values.append(block.frequencies)
        attention=block.attention_weights
        if (not 1<=attention['attn_sink'].numel()<=4096 or not 1<=attention['kv_norm.weight'].numel()<=4096
                or not 1<=attention['q_norm.weight'].numel()<=2**21 or not 1<=block.window<=2**20
                or not 1<=attention['wo_a.weight'].shape[0]//block.groups<=2**21):
            raise ValueError('DeepSeek attention exceeds explicit CUDA geometry')
        experts=(*block.packed_experts,block.packed_shared);count=len(block.packed_experts)
        if (not 2<=count<=4096 or type(block.workspace_bytes) is not int or not 1<=block.workspace_bytes<=workspace_bytes//4
                or type(block.activation_limit) not in (int,float) or not 0<block.activation_limit<=torch.finfo(torch.float32).max
                or any(e.workspace_bytes>block.workspace_bytes or any(m.format not in ('e2m1','e4m3fn') for m in (e.gate,e.up,e.down)) for e in experts)):
            raise ValueError('DeepSeek CUDA expert policy/workspace mismatch')
        table=block.weights.get('ffn.gate.tid2eid')
        if table is not None and not 1<=table.shape[0]<=2**21:raise ValueError('DeepSeek token table exceeds CUDA bounds')
        inner=_inner_workspace(block)
        if inner>workspace_bytes//4:raise ValueError('DeepSeek CUDA routing preparation exceeds workspace')
        if any(isinstance(v,PackedProjection) and v.options['workspace_bytes']>inner for v in block.attention_weights.values()):
            raise ValueError('DeepSeek CUDA attention projection exceeds workspace')
    if any(value.dtype!=torch.float32 for value in values):
        raise ValueError('explicit DeepSeek CUDA placement requires FP32 prepared tensors/projections')


def _build(source,copied,*,cuda,workspace_bytes,max_output_bytes):
    projection_type=CudaPackedProjection if cuda else PackedProjection
    matrix_type=CudaPackedMatrix if cuda else PackedMatrix
    expert_type=CudaPackedExpert if cuda else PackedExpert
    block_type=CudaDeepSeekBlock if cuda else FloatingBlock
    model_type=CudaDeepSeekModel if cuda else FloatingModel
    experts={}
    def projection(value):
        if isinstance(value,PackedProjection):
            return projection_type(copied[id(value.weight)],copied[id(value.scales)],format=value.format,
                dtype=value.dtype,row_range=(value.start,value.stop),**value.options)
        return copied[id(value)]
    def expert(value):
        if id(value) not in experts:
            matrices=[matrix_type(copied[id(m.weight)],copied[id(m.scales)],m.format) for m in (value.gate,value.up,value.down)]
            experts[id(value)]=expert_type(*matrices,row_chunk=value.options['row_chunk'],column_chunk=value.options['column_chunk'],
                workspace_bytes=value.workspace_bytes,activation_limit=value.activation_limit)
        return experts[id(value)]
    blocks=[]
    for block in source.blocks:
        options={name:getattr(block,name) for name in ('groups','window','ratio','hc_mult','top_k','route_scale','index_top_k',
                  'epsilon','connection_epsilon','iterations','activation_limit','workspace_bytes')}
        if cuda:options.update(workspace_bytes=workspace_bytes//2,max_output_bytes=workspace_bytes//2)
        blocks.append(block_type({k:projection(v) for k,v in block.attention_weights.items()},
            {k:copied[id(v)] for k,v in block.weights.items()},copied[id(block.frequencies)],
            packed_experts=tuple(expert(e) for e in block.packed_experts),packed_shared=expert(block.packed_shared),**options))
    options=dict(epsilon=source.epsilon,connection_epsilon=source.connection_epsilon)
    if cuda:options.update(workspace_bytes=workspace_bytes,max_output_bytes=max_output_bytes)
    result=model_type(copied[id(source.embedding)],blocks,{k:copied[id(v)] for k,v in source.head.items()},**options)
    if hasattr(source,'config'):
        if not isinstance(source.config,dict):raise ValueError('prepared DeepSeek config must be a mapping')
        result.config=deepcopy(source.config)
    return result


@torch.no_grad()
def place_cuda_deepseek(source,*,device,max_bytes,workspace_bytes=256*1024**2,max_output_bytes=64*1024**2):
    """Copy unique tensor objects; preserve expert aliases and packed row views.

    Validate CPU composition and CUDA metadata/policies before the first copy.
    Raw bytes exclude allocator rounding, caches, outputs and scratch. Prepared
    sources must stay immutable. Payload numerical validity is checked by the
    execution primitives; placement alone is not a numerical qualification.
    """
    device=torch.device(device)
    if device.type!='cuda' or device.index is None or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('DeepSeek placement requires indexed CUDA device and positive payload budget')
    tensors=_tensors(source);needed=sum(v.numel()*v.element_size() for v in tensors.values())
    if needed>max_bytes:raise ValueError('prepared DeepSeek exceeds CUDA raw payload budget')
    _policy(source,workspace_bytes,max_output_bytes)
    _build(source,tensors,cuda=False,workspace_bytes=workspace_bytes,max_output_bytes=max_output_bytes)
    copied={key:_copy(value,device) for key,value in tensors.items()}
    return PlacedCudaDeepSeekModel(_build(source,copied,cuda=True,workspace_bytes=workspace_bytes,max_output_bytes=max_output_bytes),needed,device)
