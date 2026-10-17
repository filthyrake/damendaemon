"""Device-resident expert routing with the pinned CPU selection semantics.

Semantics mirror ``damendaemon.expert_selection.select_experts`` (softmax, sigmoid and
sqrtsoftplus scoring, contiguous grouped selection, selection-only correction
bias, optional renormalization and scale). Inputs must be finite; validation is
identical. Everything computes on ``logits``' device with no host readback.

Device tie contract (documented deviation from unspecified native topk order):
descending stable sort keeps the original expert-ID order among equal scores,
so all paths prefer lower expert IDs on ties deterministically, including the
plain top-k path.
"""
import torch


def _policy(logits, top_k, group_count, groups_to_keep, scale):
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
            or type(scale) not in (int, float) or not 0 < scale <= torch.finfo(torch.float32).max):
        raise ValueError('invalid expert selection policy')
    return experts


def select_experts_device(logits, top_k, *, scoring='softmax', renormalize=False,
                          group_count=1, groups_to_keep=None, correction_bias=None,
                          scale=1.0, selected_experts=None, max_workspace_bytes=64*1024**2):
    """Return device weights [tokens, top_k] and int64 expert IDs without host sync.

    Scratch is bounded before allocation: the probabilities matrix, one grouped
    candidate copy and one int64 index matrix per routed token row.
    """
    if (type(renormalize) is not bool or scoring not in ('softmax', 'sigmoid', 'sqrtsoftplus')):
        raise ValueError('invalid expert selection policy')
    experts = _policy(logits, top_k, group_count, groups_to_keep, scale)
    width = experts // group_count
    if groups_to_keep is None:
        groups_to_keep = group_count
    if correction_bias is not None:
        if (not isinstance(correction_bias, torch.Tensor) or correction_bias.shape != (experts,)
                or not correction_bias.is_floating_point() or correction_bias.device != logits.device
                or width < 2):
            raise ValueError('correction bias requires one floating-point value per expert on the same device and groups of at least two')
    if not logits.isfinite().all():
        raise RuntimeError('nonfinite expert routing logits')
    tokens = logits.shape[0]
    rows = max(tokens, 1)
    scratch = 4*rows*experts*(2 if group_count > 1 else 1) + 8*rows*experts
    if type(max_workspace_bytes) is not int or not 1 <= max_workspace_bytes <= 2**31 or scratch > max_workspace_bytes:
        raise ValueError('expert routing workspace budget exceeded')
    if scoring == 'softmax':
        probabilities = torch.softmax(logits, -1, dtype=torch.float32)
    elif scoring == 'sigmoid':
        probabilities = logits.float().sigmoid()
    else:
        probabilities = torch.nn.functional.softplus(logits.float()).sqrt()
    if selected_experts is not None:
        # Preselected routes keep their supplied order (DeepSeek token tables).
        if (not isinstance(selected_experts, torch.Tensor)
                or selected_experts.shape != (tokens, top_k)
                or selected_experts.dtype not in (torch.int32, torch.int64)
                or selected_experts.device != logits.device
                or correction_bias is not None or group_count != 1):
            raise ValueError('preselected routes require matching integer rows on the same device without grouping or bias')
        selected = selected_experts.to(torch.int64)
        weights = probabilities.gather(-1, selected)
    elif group_count == 1 and correction_bias is None:
        # Descending stable sort: equal scores keep expert-ID order, so ties
        # deterministically prefer lower IDs (documented device deviation from
        # the CPU native-topk tie behavior).
        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        selected = order[:, :top_k]
        weights = probabilities.gather(-1, selected)
    else:
        scores = probabilities if correction_bias is None else probabilities + correction_bias.float()
        grouped = scores.reshape(tokens, group_count, width)
        ranks = (grouped.amax(-1) if correction_bias is None
                 else grouped.topk(2, dim=-1).values.sum(-1))
        allowed = torch.argsort(ranks, dim=-1, descending=True, stable=True)[:, :groups_to_keep]
        eligible = torch.zeros_like(ranks, dtype=torch.bool)
        eligible.scatter_(1, allowed, True)
        candidates = scores.masked_fill(~eligible.repeat_interleave(width, dim=-1), -torch.inf)
        selected = torch.argsort(candidates, dim=-1, descending=True, stable=True)[:, :top_k]
        weights = probabilities.gather(-1, selected)
    if renormalize:
        # Extremely negative non-softmax inputs can underflow every selected
        # score; preserve zero weights instead of manufacturing NaNs.
        denominator = weights.sum(-1, keepdim=True)
        if scoring != 'softmax':
            denominator = denominator.clamp_min(torch.finfo(torch.float32).tiny)
        weights = weights / denominator
    return (weights if scale == 1.0 else weights*scale), selected
