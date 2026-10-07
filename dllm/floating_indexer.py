"""Floating indexer projection/rotation/QAT composition; no packed GEMM claim."""
import torch
from .prepared_projection import is_projection, linear

from .activation_simulation import normalized_hadamard, simulate_fp4
from .compressed_indexer import select_compressed
from .deepseek_position import rotate_pairs


@torch.no_grad()
def index_hidden(hidden, low_rank, query_weight, head_weight, keys, positions, inverse_frequencies,
                 *, top_k, offset=0, query_chunk=16, key_chunk=64):
    """Select ratio-four compressed blocks using already resident floating weights.

    hidden [batch,sequence,model_width], low_rank [batch,sequence,query_rank],
    query_weight [heads*index_width,query_rank], head_weight [heads,model_width],
    keys [batch,blocks,index_width]. All use the same FP16/BF16/FP32 dtype/device.
    Projection matrices may instead be explicit CPU packed adapters with matching
    result dtype. Keys must already have the compressor's rotation/FP4 transform.
    Frequency and position contracts match rotate_pairs. Accelerator FP8 activation rounding and performance remain unqualified.
    """
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] < 1
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('indexer hidden input must be floating [batch,sequence,width]')
    for tensor, rank in ((low_rank, 3), (keys, 3)):
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != rank
                or tensor.dtype != hidden.dtype or tensor.device != hidden.device):
            raise ValueError('floating indexer inputs/weights must match hidden dtype and device')
    for weight in (query_weight,head_weight):
        if not is_projection(weight) or weight.dtype != hidden.dtype or weight.device != hidden.device:
            raise ValueError('indexer projections must match hidden dtype and device')
    width, heads = keys.shape[-1], head_weight.shape[0]
    if (not 32 <= width <= 4096 or width & (width-1) or heads < 1 or low_rank.shape[-1] < 1
            or low_rank.shape[:2] != hidden.shape[:2] or keys.shape[0] != hidden.shape[0]
            or head_weight.shape[1] != hidden.shape[-1]
            or query_weight.shape != (heads*width, low_rank.shape[-1])):
        raise ValueError('floating indexer projection geometry does not agree')
    if (not isinstance(positions, torch.Tensor) or positions.shape != hidden.shape[:2]
            or positions.dtype != torch.int64 or positions.device != hidden.device
            or not isinstance(inverse_frequencies, torch.Tensor) or inverse_frequencies.ndim != 1
            or inverse_frequencies.dtype != torch.float32 or inverse_frequencies.device != hidden.device
            or not 1 <= inverse_frequencies.numel() <= width//2):
        raise ValueError('invalid indexer rotary positions or frequencies')
    if (type(top_k) is not int or not 1 <= top_k <= 4096
            or type(offset) is not int or not 0 <= offset < 2**31
            or type(query_chunk) is not int or not 1 <= query_chunk <= 128
            or type(key_chunk) is not int or not 1 <= key_chunk <= 256):
        raise ValueError('invalid indexer selection/chunk options')
    count = min(top_k, keys.shape[1])
    ids = torch.empty(*hidden.shape[:2], count, device=hidden.device, dtype=torch.int64)
    scores = torch.empty(*hidden.shape[:2], count, device=hidden.device, dtype=torch.float32)
    rotary_width = inverse_frequencies.numel()*2
    for start in range(0, hidden.shape[1], query_chunk):
        stop = min(start+query_chunk, hidden.shape[1])
        queries = linear(low_rank[:, start:stop], query_weight).unflatten(-1, (heads, width))
        queries[..., -rotary_width:] = rotate_pairs(queries[..., -rotary_width:], positions[:, start:stop], inverse_frequencies)
        queries = simulate_fp4(normalized_hadamard(queries))
        # Preserve the reference BF16/FP16 coefficient rounding before FP32 scoring.
        weights = (linear(hidden[:, start:stop], head_weight)*(width**-.5*heads**-.5)).float()
        selected, selected_scores = select_compressed(queries, keys, weights, positions[:, start:stop],
                                                     ratio=4, top_k=top_k, offset=offset, query_chunk=query_chunk,
                                                     key_chunk=key_chunk, scale_head_weights=False)
        ids[:, start:stop], scores[:, start:stop] = selected, selected_scores
    return ids, scores
