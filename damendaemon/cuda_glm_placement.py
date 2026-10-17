"""Explicit whole prepared packed GLM placement with pre-copy payload admission."""
from dataclasses import dataclass
from copy import deepcopy
import torch
from .glm_model import GlmModel
from .glm_layer_stage import GlmLayerStage
from .cuda_layer_stages import CudaGlmLayerStage
from .glm_block import GlmBlock
from .glm_cached_attention import CachedGlmAttention
from .glm_packed_ffn import PackedGlmFeedForward
from .packed_moe import PackedExpert,PackedMatrix
from .prepared_projection import PackedProjection
from .cuda_glm_model import CudaGlmModel,CudaGlmBlock
from .cuda_glm_cached_attention import CudaCachedGlmAttention
from .cuda_glm_ffn import CudaPackedGlmFeedForward
from .cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
from .cuda_prepared_projection import CudaPackedProjection
from .cuda_packed_placement import _copy


@dataclass(frozen=True)
class PlacedCudaGlmModel:
    model: CudaGlmModel
    raw_bytes: int
    device: torch.device


def _tensors(source,*,stage=False):
    if not isinstance(source,GlmLayerStage if stage else GlmModel):
        raise ValueError('placement requires a prepared CPU GLM model')
    tensors={}
    def add(value):
        if isinstance(value,PackedProjection):
            add(value.weight);add(value.scales)
        elif isinstance(value,torch.Tensor) and value.device.type=='cpu':
            tensors[id(value)]=value
        else:raise ValueError('GLM placement requires CPU tensors or packed projections')
    for value in (source.embedding,source.norm,source.head):
        if value is not None or not stage:add(value)
    for block in source.blocks:
        if not isinstance(block,GlmBlock) or not isinstance(block.attention,CachedGlmAttention) or not isinstance(block.feed_forward,PackedGlmFeedForward):
            raise ValueError('placement requires prepared CPU packed GLM components')
        add(block.input_norm);add(block.post_norm)
        for value in block.attention.weights.values():add(value)
        ffn=block.feed_forward
        if ffn.kind=='sparse':add(ffn.router);add(ffn.correction_bias)
        for expert in ffn.experts+(() if ffn.shared is None else (ffn.shared,)):
            if not isinstance(expert,PackedExpert):raise ValueError('GLM placement requires CPU packed experts')
            for matrix in (expert.gate,expert.up,expert.down):add(matrix.weight);add(matrix.scales)
    return tensors


def _build(source,copied,*,cuda,stage=False):
    """Revalidate the full CPU composition before using the same recipe on CUDA."""
    projection_type=CudaPackedProjection if cuda else PackedProjection
    matrix_type=CudaPackedMatrix if cuda else PackedMatrix
    expert_type=CudaPackedExpert if cuda else PackedExpert
    attention_type=CudaCachedGlmAttention if cuda else CachedGlmAttention
    ffn_type=CudaPackedGlmFeedForward if cuda else PackedGlmFeedForward
    block_type=CudaGlmBlock if cuda else GlmBlock
    model_type=CudaGlmModel if cuda else GlmModel
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
        a=block.attention;f=block.feed_forward
        attention=attention_type({name:projection(w) for name,w in a.weights.items()},heads=a.heads,nope_dim=a.nope_dim,
            rope_dim=a.rope_dim,value_dim=a.value_dim,index_top_k=a.index_top_k,indexer_type='full' if a.indexer is not None else 'shared',
            rope_base=a.rope_base,epsilon=a.epsilon,**a.budgets)
        bank=tuple(expert(e) for e in f.experts)
        options=dict(activation_limit=f.activation_limit,max_output_bytes=f.max_output_bytes,workspace_bytes=f.workspace_bytes)
        if f.kind=='sparse':options.update(router=copied[id(f.router)],correction_bias=copied[id(f.correction_bias)],shared=expert(f.shared),top_k=f.top_k,route_scale=f.route_scale)
        feed_forward=ffn_type(bank[0] if f.kind=='dense' else bank,**options)
        blocks.append(block_type(attention,feed_forward,copied[id(block.input_norm)],copied[id(block.post_norm)],epsilon=block.epsilon))
    if stage:
        kind=CudaGlmLayerStage if cuda else GlmLayerStage
        model=kind(blocks,start=source.start,total=source.total,
            embedding=None if source.embedding is None else copied[id(source.embedding)],
            norm=None if source.norm is None else copied[id(source.norm)],head=None if source.head is None else copied[id(source.head)],
            epsilon=source.epsilon,max_output_bytes=source.max_output_bytes)
    else:
        model=model_type(copied[id(source.embedding)],blocks,copied[id(source.norm)],copied[id(source.head)],
                         epsilon=source.epsilon,max_output_bytes=source.max_output_bytes)
    if hasattr(source,'config'):
        if not isinstance(source.config,dict):raise ValueError('prepared GLM config must be a mapping')
        model.config=deepcopy(source.config)
    return model


@torch.no_grad()
def place_cuda_glm(source,*,device,max_bytes):
    """Admit all raw payloads and CPU metadata before copying any tensor.

    Sources must remain immutable during placement. Identical tensor objects
    share one compact CUDA copy; distinct views count separately. The byte cap
    excludes allocator rounding, scratch, caches and outputs; callers separately
    admit those resources. No cache is allocated or serving policy changed.
    """
    device=torch.device(device)
    if device.type!='cuda' or device.index is None or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('GLM placement requires indexed CUDA device and positive byte budget')
    tensors=_tensors(source)
    needed=sum(t.numel()*t.element_size() for t in tensors.values())
    if needed>max_bytes:raise ValueError('prepared GLM exceeds CUDA raw payload budget')
    _build(source,tensors,cuda=False)
    copied={key:_copy(value,device) for key,value in tensors.items()}
    return PlacedCudaGlmModel(_build(source,copied,cuda=True),needed,device)
