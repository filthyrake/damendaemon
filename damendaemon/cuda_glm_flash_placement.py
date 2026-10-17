"""Whole prepared packed Flash placement with complete pre-copy admission."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from .glm_flash_model import FlashModel
from .glm_flash_layer_stage import FlashLayerStage
from .cuda_layer_stages import CudaFlashLayerStage
from .glm_flash_block import FlashBlock
from .glm_flash_linear import FlashLinearAttention
from .glm_flash_sparse import FlashSparseAttention
from .glm_flash_indexer import FlashIndexer
from .glm_flash_hyper import FlashHyperConnection
from .glm_flash_ffn import PackedFlashFeedForward
from .packed_moe import PackedExpert,PackedMatrix
from .prepared_projection import PackedProjection
from .cuda_glm_flash_model import CudaFlashModel,CudaFlashBlock,CudaPackedFlashFeedForward
from .cuda_glm_flash_linear import CudaFlashLinearAttention
from .cuda_glm_flash_sparse import CudaFlashSparseAttention
from .cuda_glm_flash_indexer import CudaFlashIndexer
from .cuda_glm_flash_primitives import CudaFlashHyperConnection
from .cuda_packed_expert import CudaPackedExpert,CudaPackedMatrix
from .cuda_prepared_projection import CudaPackedProjection
from .cuda_packed_placement import _copy


@dataclass(frozen=True)
class PlacedCudaFlashModel:
    model: CudaFlashModel
    raw_bytes: int
    device: torch.device


def _tensors(source,*,stage=False):
    if not isinstance(source,FlashLayerStage if stage else FlashModel):raise ValueError('placement requires a prepared CPU Flash model')
    tensors={}
    def add(value):
        if isinstance(value,PackedProjection):add(value.weight);add(value.scales)
        elif isinstance(value,torch.Tensor) and value.device.type=='cpu':tensors[id(value)]=value
        else:raise ValueError('Flash placement requires CPU tensors or packed projections')
    for value in (source.embedding,source.norm,source.head):
        if value is not None or not stage:add(value)
    for block in source.blocks:
        if (not isinstance(block,FlashBlock) or not isinstance(block.attention,(FlashLinearAttention,FlashSparseAttention))
                or not isinstance(block.feed_forward,PackedFlashFeedForward)):
            raise ValueError('placement requires prepared CPU packed Flash components')
        add(block.input_norm);add(block.post_norm)
        for h in (block.attention_hyper,block.ffn_hyper):
            if not isinstance(h,FlashHyperConnection):raise ValueError('invalid Flash hyper component')
            for v in (h.projection,h.base,h.scale):add(v)
        a=block.attention
        for v in a.weights.values():add(v)
        if isinstance(a,FlashSparseAttention):
            if not isinstance(a.indexer,FlashIndexer):raise ValueError('invalid Flash indexer')
            for v in a.indexer.weights.values():add(v)
        f=block.feed_forward
        if f.kind=='sparse':add(f.router);add(f.correction_bias)
        for e in f.experts+(() if f.shared is None else (f.shared,)):
            if not isinstance(e,PackedExpert):raise ValueError('Flash placement requires CPU packed experts')
            for m in (e.gate,e.up,e.down):add(m.weight);add(m.scales)
    return tensors


def _build(source,copied,*,cuda,stage=False,device_routing=False,composition='per_token'):
    projection_type=CudaPackedProjection if cuda else PackedProjection
    matrix_type=CudaPackedMatrix if cuda else PackedMatrix
    expert_type=CudaPackedExpert if cuda else PackedExpert
    ffn_type=CudaPackedFlashFeedForward if cuda else PackedFlashFeedForward
    linear_type=CudaFlashLinearAttention if cuda else FlashLinearAttention
    sparse_type=CudaFlashSparseAttention if cuda else FlashSparseAttention
    index_type=CudaFlashIndexer if cuda else FlashIndexer
    hyper_type=CudaFlashHyperConnection if cuda else FlashHyperConnection
    block_type=CudaFlashBlock if cuda else FlashBlock
    model_type=CudaFlashModel if cuda else FlashModel
    experts={}
    def limits(c):return dict(max_workspace_bytes=c.max_workspace_bytes,max_output_bytes=c.max_output_bytes)
    def projection(v):
        if isinstance(v,PackedProjection):
            return projection_type(copied[id(v.weight)],copied[id(v.scales)],format=v.format,dtype=v.dtype,row_range=(v.start,v.stop),**v.options)
        return copied[id(v)]
    def expert(e):
        if id(e) not in experts:
            matrices=[matrix_type(copied[id(m.weight)],copied[id(m.scales)],m.format) for m in (e.gate,e.up,e.down)]
            experts[id(e)]=expert_type(*matrices,row_chunk=e.options['row_chunk'],column_chunk=e.options['column_chunk'],workspace_bytes=e.workspace_bytes,activation_limit=e.activation_limit)
        return experts[id(e)]
    def hyper(h):
        return hyper_type(copied[id(h.projection)],copied[id(h.base)],copied[id(h.scale)],streams=h.streams,
                          norm_epsilon=h.norm_epsilon,connection_epsilon=h.connection_epsilon,iterations=h.iterations,**limits(h))
    blocks=[]
    for block in source.blocks:
        a=block.attention;f=block.feed_forward;weights={n:projection(v) for n,v in a.weights.items()}
        if isinstance(a,FlashLinearAttention):
            attention=linear_type(weights,heads=a.heads,head_dim=a.head_dim,kernel_size=a.kernel_size,lower_bound=a.lower_bound,epsilon=a.epsilon,**limits(a))
        else:
            i=a.indexer
            index=index_type({n:copied[id(v)] for n,v in i.weights.items()},heads=i.heads,pool_size=i.pool_size,top_k=i.top_k,**limits(i))
            attention=sparse_type(weights,index,heads=a.heads,head_dim=a.head_dim,value_dim=a.value_dim,epsilon=a.epsilon,**limits(a))
        bank=tuple(expert(e) for e in f.experts)
        options=dict(activation_limit=f.activation_limit,max_output_bytes=f.max_output_bytes,workspace_bytes=f.workspace_bytes)
        if cuda:options.update(device_routing=device_routing,composition=composition)
        if f.kind=='sparse':options.update(router=copied[id(f.router)],correction_bias=copied[id(f.correction_bias)],shared=expert(f.shared),top_k=f.top_k,route_scale=f.route_scale)
        feed=ffn_type(bank[0] if f.kind=='dense' else bank,**options)
        blocks.append(block_type(attention,feed,hyper(block.attention_hyper),hyper(block.ffn_hyper),copied[id(block.input_norm)],copied[id(block.post_norm)],epsilon=block.epsilon,**limits(block)))
    if stage:
        kind=CudaFlashLayerStage if cuda else FlashLayerStage
        result=kind(blocks,start=source.start,total=source.total,
            embedding=None if source.embedding is None else copied[id(source.embedding)],
            norm=None if source.norm is None else copied[id(source.norm)],head=None if source.head is None else copied[id(source.head)],
            epsilon=source.epsilon,max_output_bytes=source.max_output_bytes)
    else:
        result=model_type(copied[id(source.embedding)],blocks,copied[id(source.norm)],copied[id(source.head)],epsilon=source.epsilon,max_output_bytes=source.max_output_bytes)
    if hasattr(source,'config'):
        if not isinstance(source.config,dict):raise ValueError('prepared Flash config must be a mapping')
        result.config=deepcopy(source.config)
    return result


@torch.no_grad()
def place_cuda_flash(source,*,device,max_bytes,device_routing=False,composition='per_token'):
    """Validate all CPU metadata and payloads before copying. Sources stay immutable.

    Identical tensor objects share a compact copy; distinct views count separately.
    Raw payload excludes allocator rounding, outputs, caches and scratch, which
    callers must admit separately. No default dispatch or serving policy changes.
    Composition and device routing are opt-in FFN dispatch options; the CPU
    revalidation build always runs on defaults.
    """
    device=torch.device(device)
    if device.type!='cuda' or device.index is None or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('Flash placement requires indexed CUDA device and positive byte budget')
    if type(device_routing) is not bool:raise ValueError('CUDA Flash device routing must be boolean')
    if composition not in ('per_token','batched'):
        raise ValueError("CUDA Flash composition must be 'per_token' or 'batched'")
    tensors=_tensors(source);needed=sum(v.numel()*v.element_size() for v in tensors.values())
    if needed>max_bytes:raise ValueError('prepared Flash exceeds CUDA raw payload budget')
    _build(source,tensors,cuda=False)
    copied={key:_copy(value,device) for key,value in tensors.items()}
    return PlacedCudaFlashModel(_build(source,copied,cuda=True,device_routing=device_routing,
                                       composition=composition),needed,device)
