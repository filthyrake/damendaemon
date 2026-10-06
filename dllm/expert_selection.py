"""Device-tensor expert routing foundations; no architecture qualification.

Independent implementation of grouped expert-selection semantics. Reference:
vLLM grouped_topk_router.py at 84bcbc62644356270aaaa5e2d0237d03adc9bb3a
(Apache-2.0). No upstream runtime is imported. Inputs must be finite.
Token-table/sqrtsoftplus semantics: DeepSeek-V4-Flash inference/model.py at
60d8d70770c6776ff598c94bb586a859a38244f1 (MIT).
"""
import torch


def select_experts(logits, top_k, *, scoring='softmax', renormalize=False,
                   group_count=1, groups_to_keep=None, correction_bias=None, scale=1.0,
                   selected_experts=None):
    """Return FP32 weights and int64 expert IDs, without host routing decisions.

    Groups are contiguous. Unbiased group ranking uses the maximum probability;
    biased ranking uses the sum of its two highest adjusted scores. Bias affects
    selection only, never mixture weights. Grouped/biased ties prefer lower IDs.
    The plain softmax path preserves the existing native topk ordering.
    Preselected IDs must already be validated; their supplied order is retained.
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
            or type(renormalize) is not bool or scoring not in ('softmax', 'sigmoid', 'sqrtsoftplus')
            or type(scale) not in (int, float) or not 0 < scale <= torch.finfo(torch.float32).max):
        raise ValueError('invalid expert selection policy')
    width = experts // group_count
    if correction_bias is not None:
        if (not isinstance(correction_bias, torch.Tensor) or correction_bias.shape != (experts,) or not correction_bias.is_floating_point()
                or correction_bias.device != logits.device or width < 2):
            raise ValueError('correction bias requires one floating-point value per expert on the same device and groups of at least two')
    if selected_experts is not None:
        if (not isinstance(selected_experts, torch.Tensor)
                or selected_experts.shape != (logits.shape[0], top_k)
                or selected_experts.dtype not in (torch.int32, torch.int64)
                or selected_experts.device != logits.device
                or correction_bias is not None or group_count != 1):
            raise ValueError('preselected routes require matching integer rows on the same device without grouping or bias')
    if scoring == 'softmax':
        probabilities = torch.softmax(logits, -1, dtype=torch.float32)
    elif scoring == 'sigmoid':
        probabilities = logits.float().sigmoid()
    else:
        probabilities = torch.nn.functional.softplus(logits.float()).sqrt()
    if selected_experts is not None:
        selected = selected_experts.to(torch.int64)
        weights = probabilities.gather(-1, selected)
    elif group_count == 1 and correction_bias is None:
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
        # Extremely negative non-softmax inputs can underflow every selected score.
        # Preserve zero weights instead of manufacturing NaNs or uniform routes.
        denominator = weights.sum(-1, keepdim=True)
        if scoring != 'softmax':
            denominator = denominator.clamp_min(torch.finfo(torch.float32).tiny)
        weights = weights / denominator
    return (weights if scale == 1.0 else weights * scale), selected


def validate_token_expert_table(table, vocab_size, experts, top_k):
    """Validate checkpoint table values on CPU once, before device placement."""
    if (any(type(value) is not int or value < 1 for value in (vocab_size, experts, top_k))
            or top_k > experts or not isinstance(table, torch.Tensor)
            or table.device.type != 'cpu' or table.dtype not in (torch.int32, torch.int64)
            or table.shape != (vocab_size, top_k)):
        raise ValueError('token expert table must match vocabulary/top-k as a CPU integer matrix')
    if ((table < 0).any() or (table >= experts).any()
            or (table.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError('token expert table contains out-of-range or duplicate expert IDs')


def select_token_experts(logits, token_ids, table, *, scoring='sqrtsoftplus',
                         renormalize=True, scale=1.0):
    """Route using a prevalidated checkpoint table; no host decisions per token.

    Call validate_token_expert_table before placing the table on a device. Token
    IDs must already have passed the serving vocabulary bounds check.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or not isinstance(token_ids, torch.Tensor) or token_ids.shape != (logits.shape[0],)
            or token_ids.dtype not in (torch.int32, torch.int64)
            or not isinstance(table, torch.Tensor) or table.ndim != 2
            or table.dtype not in (torch.int32, torch.int64)
            or token_ids.device != logits.device or table.device != logits.device):
        raise ValueError('token routing needs integer token IDs and a table on the logits device')
    selected = table.index_select(0, token_ids.to(torch.int64))
    return select_experts(logits, table.shape[1], scoring=scoring, renormalize=renormalize,
                          scale=scale, selected_experts=selected)
