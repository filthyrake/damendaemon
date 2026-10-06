"""Device-tensor expert routing foundations; no architecture qualification.

Independent implementation of grouped expert-selection semantics. Reference:
vLLM grouped_topk_router.py at 84bcbc62644356270aaaa5e2d0237d03adc9bb3a
(Apache-2.0). No upstream runtime is imported. Inputs must be finite.
"""
import torch


def select_experts(logits, top_k, *, scoring='softmax', renormalize=False,
                   group_count=1, groups_to_keep=None, correction_bias=None, scale=1.0):
    """Return FP32 weights and int64 expert IDs, without host routing decisions.

    Groups are contiguous. Unbiased group ranking uses the maximum probability;
    biased ranking uses the sum of its two highest adjusted scores. Bias affects
    selection only, never mixture weights. Grouped/biased ties prefer lower IDs.
    The plain softmax path preserves the existing native topk ordering.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not logits.is_floating_point() or logits.shape[1] < 1:
        raise ValueError('expert logits must be a floating-point token/expert matrix')
    experts = logits.shape[1]
    if groups_to_keep is None:
        groups_to_keep = group_count
    if (type(top_k) is not int or not 1 <= top_k <= experts
            or type(group_count) is not int or not 1 <= group_count <= experts
            or experts % group_count or type(groups_to_keep) is not int
            or not 1 <= groups_to_keep <= group_count
            or top_k > groups_to_keep * (experts // group_count)
            or type(renormalize) is not bool or scoring not in ('softmax', 'sigmoid')
            or type(scale) not in (int, float) or not 0 < scale <= torch.finfo(torch.float32).max):
        raise ValueError('invalid expert selection policy')
    width = experts // group_count
    if correction_bias is not None:
        if (not isinstance(correction_bias, torch.Tensor) or correction_bias.shape != (experts,) or not correction_bias.is_floating_point()
                or correction_bias.device != logits.device or width < 2):
            raise ValueError('correction bias requires one floating-point value per expert on the same device and groups of at least two')
    probabilities = (torch.softmax(logits, -1, dtype=torch.float32) if scoring == 'softmax'
                     else logits.float().sigmoid())
    if group_count == 1 and correction_bias is None:
        weights, selected = probabilities.topk(top_k, dim=-1)
    else:
        scores = probabilities if correction_bias is None else probabilities + correction_bias.float()
        grouped = scores.reshape(logits.shape[0], group_count, width)
        ranks = (grouped.amax(-1) if correction_bias is None
                 else grouped.topk(2, dim=-1).values.sum(-1))
        allowed = torch.argsort(ranks, dim=-1, descending=True, stable=True)[:, :groups_to_keep]
        eligible = torch.zeros_like(ranks, dtype=torch.bool)
        eligible.scatter_(1, allowed, True)
        candidates = scores.masked_fill(~eligible.repeat_interleave(width, dim=-1), -torch.inf)
        selected = torch.argsort(candidates, dim=-1, descending=True, stable=True)[:, :top_k]
        weights = probabilities.gather(-1, selected)
    if renormalize:
        # Extremely negative sigmoid inputs can underflow every selected score.
        # Preserve zero weights instead of manufacturing NaNs or uniform routes.
        denominator = weights.sum(-1, keepdim=True)
        if scoring == 'sigmoid':
            denominator = denominator.clamp_min(torch.finfo(torch.float32).tiny)
        weights = weights / denominator
    return (weights if scale == 1.0 else weights * scale), selected
