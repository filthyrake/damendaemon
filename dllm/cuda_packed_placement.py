"""Explicit budgeted placement of CPU-prepared packed experts; no model dispatch."""
from dataclasses import dataclass
import torch

from .packed_moe import PackedExpert
from .cuda_packed_expert import CudaPackedExpert, CudaPackedMatrix


@dataclass(frozen=True)
class CudaExpertBank:
    experts: tuple
    shared: CudaPackedExpert | None
    raw_bytes: int
    device: torch.device


def _options(expert):
    return dict(row_chunk=expert.options['row_chunk'],column_chunk=expert.options['column_chunk'],
                workspace_bytes=expert.workspace_bytes,activation_limit=expert.activation_limit)


def _plan(experts, shared, max_bytes):
    if (not isinstance(experts,(list,tuple)) or not experts
            or type(max_bytes) is not int or max_bytes < 1):
        raise ValueError('placement requires a nonempty CPU bank and positive byte budget')
    all_experts = tuple(experts)+( () if shared is None else (shared,) )
    tensors = {}
    for expert in all_experts:
        if not isinstance(expert,PackedExpert):
            raise ValueError('placement requires prepared CPU packed experts')
        # Prepared objects are mutable: revalidate every member before any copy.
        PackedExpert(expert.gate,expert.up,expert.down,**_options(expert))
        if expert.width != experts[0].width or expert.gate.shape[1] != expert.width:
            raise ValueError('placement bank widths disagree')
        for matrix in (expert.gate,expert.up,expert.down):
            for tensor in (matrix.weight,matrix.scales):
                tensors[id(tensor)] = tensor
    raw_bytes = sum(t.numel()*t.element_size() for t in tensors.values())
    if raw_bytes > max_bytes:
        raise ValueError('packed CUDA bank exceeds resident payload budget')
    return all_experts,tensors,raw_bytes


def _copy(tensor,device):
    # A compact owned copy makes logical payload accounting exact for strided
    # views. Identical tensor objects are shared; distinct views count separately.
    return tensor.detach().contiguous().to(device=device,copy=True)


@torch.no_grad()
def place_cuda_experts(experts, *, device, max_bytes, shared=None):
    """Copy an admitted packed bank, preserving tensor-object aliases.

    max_bytes bounds raw weight/scale payload only. Device allocator rounding,
    scratch, route buffers and outputs require separate admission. Source tensors
    must not be mutated during placement. Copies are owned by the returned bank;
    on failure no bank is published and no shared CUDA cache is cleared.
    """
    device = torch.device(device)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('placement requires an explicit indexed CUDA device')
    all_experts,tensors,raw_bytes = _plan(experts,shared,max_bytes)
    copied = {key:_copy(value,device) for key,value in tensors.items()}
    prepared = {}
    for expert in all_experts:
        if id(expert) not in prepared:
            matrices = [CudaPackedMatrix(copied[id(m.weight)],copied[id(m.scales)],m.format)
                        for m in (expert.gate,expert.up,expert.down)]
            prepared[id(expert)] = CudaPackedExpert(*matrices,**_options(expert))
    return CudaExpertBank(tuple(prepared[id(e)] for e in experts),
                          None if shared is None else prepared[id(shared)],raw_bytes,device)
