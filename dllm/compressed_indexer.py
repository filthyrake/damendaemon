"""Causal weighted-ReLU indexer equations with chunk-bounded score storage.

Independent DeepSeek-V4-Flash Indexer equations from inference/model.py (MIT),
revision 60d8d70770c6776ff598c94bb586a859a38244f1. FP32 arithmetic here does not
qualify reference QAT, Hadamard rotation, accelerator rounding or GPU performance.
"""
import torch


@torch.no_grad()
def _select_compressed(query, keys, head_weights, positions, *, ratio, top_k,
                      offset=0, query_chunk=16, key_chunk=64, scale_head_weights=True, dot_scale=1.,validate_scores=False):
    """Return physical IDs and FP32 scores, padding unavailable blocks with -1/-inf.

    query [batch,sequence,heads,width], keys [batch,blocks,width], head_weights
    FP32 [batch,sequence,heads], logical positions int64 [batch,sequence]. The
    caller validates nonnegative positions and supplies finite projected inputs.
    Blocks are visible once (position+1)//ratio input blocks have completed.
    Equal scores choose lower logical block IDs, independently of chunk sizes.
    Set scale_head_weights=False only for coefficients already scaled by
    1/sqrt(width*heads), such as storage-rounded projection outputs.
    dot_scale applies before ReLU; its default preserves V4 arithmetic.
    Only chunks of query/key tensors are promoted; resident inputs remain intact.
    """
    if (not isinstance(query, torch.Tensor) or query.ndim != 4
            or query.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or min(query.shape[-2:]) < 1):
        raise ValueError('indexer query must be floating [batch,sequence,heads,width]')
    batch, sequence, heads, width = query.shape
    if (not isinstance(keys, torch.Tensor) or keys.ndim != 3 or keys.shape[0] != batch
            or keys.shape[2] != width or keys.dtype != query.dtype or keys.device != query.device):
        raise ValueError('indexer keys must match query batch,width,dtype and device')
    if (not isinstance(head_weights, torch.Tensor) or head_weights.shape != query.shape[:3]
            or head_weights.dtype != torch.float32 or head_weights.device != query.device):
        raise ValueError('indexer head weights must be same-device FP32 [batch,sequence,heads]')
    if (not isinstance(positions, torch.Tensor) or positions.shape != query.shape[:2]
            or positions.dtype != torch.int64 or positions.device != query.device):
        raise ValueError('indexer positions must be same-device int64 [batch,sequence]')
    if (type(ratio) is not int or not 1 <= ratio <= 128
            or type(top_k) is not int or not 1 <= top_k <= 4096
            or type(offset) is not int or not 0 <= offset < 2**31
            or type(query_chunk) is not int or not 1 <= query_chunk <= 128
            or type(key_chunk) is not int or not 1 <= key_chunk <= 256
            or type(scale_head_weights) is not bool or type(dot_scale) not in (int,float)
            or not 0 < dot_scale <= torch.finfo(torch.float32).max):
        raise ValueError('invalid indexer ratio, top-k, offset or chunk bounds')
    top = min(top_k, keys.shape[1])
    selected = torch.empty(batch, sequence, top, device=query.device, dtype=torch.int64)
    selected_scores = torch.empty(batch, sequence, top, device=query.device, dtype=torch.float32)
    for start in range(0, sequence, query_chunk):
        stop = min(sequence, start+query_chunk)
        q = query[:, start:stop].float()
        weights = head_weights[:, start:stop]
        if scale_head_weights:
            weights = weights*(width**-.5*heads**-.5)
        visible = (positions[:, start:stop]+1)//ratio
        best_scores = selected_scores[:, start:stop, :0]
        best_ids = selected[:, start:stop, :0]
        for key_start in range(0, keys.shape[1], key_chunk):
            key_stop = min(key_start+key_chunk, keys.shape[1])
            key = keys[:, key_start:key_stop].float()
            dots = torch.matmul(q, key[:, None].transpose(-1, -2))
            if dot_scale != 1.:dots = dots*dot_scale
            scores = (dots.relu()*weights.unsqueeze(-1)).sum(-2)
            ids = torch.arange(key_start, key_stop, device=query.device).reshape(1, 1, -1).expand(batch, stop-start, -1)
            scores = scores.masked_fill(ids >= visible.unsqueeze(-1), -float('inf'))
            if validate_scores and ((ids < visible.unsqueeze(-1)) & ~torch.isfinite(scores)).any():
                raise ValueError('nonfinite visible compressed indexer score')
            candidates = torch.cat((best_scores, scores), -1)
            candidate_ids = torch.cat((best_ids, ids), -1)
            order = torch.argsort(candidates, descending=True, stable=True)[..., :top]
            best_scores = candidates.gather(-1, order)
            best_ids = candidate_ids.gather(-1, order)
        selected[:, start:stop] = torch.where(best_ids < visible.unsqueeze(-1), best_ids+offset, -1)
        selected_scores[:, start:stop] = best_scores
    return selected, selected_scores


def select_compressed(query,keys,head_weights,positions,**options):
    """Generic reference selection; caller-owned input validation is unchanged."""
    return _select_compressed(query,keys,head_weights,positions,**options)
