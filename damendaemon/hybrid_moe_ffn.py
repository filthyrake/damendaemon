"""Host reference for the Qwen3.5-MoE text FFN (#14 Qwen target).

Tensor structure follows the pinned checkpoint contract
(``damendaemon.hybrid_moe_checkpoint``): a router ``mlp.gate.weight``
[experts, hidden]; fused stacked experts ``mlp.experts.gate_up_proj``
[experts, 2*inter, hidden] with ``mlp.experts.down_proj`` [experts, hidden,
inter]; a dense shared expert (gate/up [shared, hidden], down [hidden,
shared]); and a [1, hidden] ``shared_expert_gate`` sigmoid combine gate.
Expert execution mirrors the sparse-MoE block of Transformers 4.56.1
qwen3_moe modeling (one pass per routed expert with weighted accumulation)
and routing reuses the audited ``damendaemon.expert_selection.select_experts``.

The qwen3_5_moe_text checkpoint config pins no routing keys (there is no
norm_topk_prob or scoring field), so scoring and renormalization are
explicit constructor policies here. The upstream qwen3_moe family block uses
softmax scoring with config-gated top-k renormalization; neither policy is
qualified against the real checkpoint - that pinning belongs to the
real-checkpoint executor cell. FP32 CPU arithmetic only; no model execution
or quality is claimed.
"""
import torch
import torch.nn.functional as F
from .expert_selection import select_experts

SCORINGS = frozenset(('softmax', 'sigmoid'))


def _matrix(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.dtype is not torch.float32
            or value.device.type != 'cpu' or tuple(value.shape) != shape
            or not torch.isfinite(value).all()):
        raise ValueError('qwen3.5-MoE FFN requires a finite CPU FP32 ' + name)


class HybridMoeFeedForward:
    """Route every token through top-k fused experts plus a gated shared expert.

    Weights are borrowed immutably. ``forward`` gathers the tokens routed to
    each expert into one pass, accumulates weighted contributions, then adds
    the shared expert scaled by the sigmoid of its combine gate. Scratch
    (routing logits, per-expert and shared intermediates) is bounded by
    ``workspace_bytes``; resident weights and the final output are excluded
    from that bound, and the output is separately bounded by
    ``max_output_bytes``.
    """

    def __init__(self, router, gate_up, down, shared_gate, shared_up, shared_down,
                 shared_expert_gate, *, top_k, scoring='sigmoid', renormalize=True,
                 max_output_bytes=64 * 1024 ** 2, workspace_bytes=64 * 1024 ** 2):
        if (type(top_k) is not int or scoring not in SCORINGS or type(renormalize) is not bool
                or type(max_output_bytes) is not int or not 0 < max_output_bytes <= 2 ** 53
                or type(workspace_bytes) is not int or not 0 < workspace_bytes <= 2 ** 53):
            raise ValueError('invalid qwen3.5-MoE FFN routing policy or byte budget')

        def _checked(value, name):
            if (not isinstance(value, torch.Tensor) or value.dtype is not torch.float32
                    or value.device.type != 'cpu' or not torch.isfinite(value).all()):
                raise ValueError('qwen3.5-MoE FFN requires a finite CPU FP32 ' + name)
            return value

        _checked(router, 'router')
        _checked(gate_up, 'fused expert gate/up projection')
        _checked(down, 'fused expert down projection')
        _checked(shared_gate, 'shared expert gate projection')
        _checked(shared_up, 'shared expert up projection')
        _checked(shared_down, 'shared expert down projection')
        _checked(shared_expert_gate, 'shared expert combine gate')
        if router.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3 or shared_gate.ndim != 2 \
                or shared_up.ndim != 2 or shared_down.ndim != 2 or shared_expert_gate.ndim != 2:
            raise ValueError('invalid qwen3.5-MoE FFN tensor ranks')
        experts, dim = router.shape
        inter = gate_up.shape[1] // 2
        shared = shared_gate.shape[0]
        if not 2 <= experts or not 1 <= dim or not 1 <= inter or not 1 <= shared or not 1 <= top_k <= experts:
            raise ValueError('invalid qwen3.5-MoE FFN geometry')
        _matrix(router, (experts, dim), 'router')
        _matrix(gate_up, (experts, 2 * inter, dim), 'fused expert gate/up projection')
        _matrix(down, (experts, dim, inter), 'fused expert down projection')
        _matrix(shared_gate, (shared, dim), 'shared expert gate projection')
        _matrix(shared_up, (shared, dim), 'shared expert up projection')
        _matrix(shared_down, (dim, shared), 'shared expert down projection')
        _matrix(shared_expert_gate, (1, dim), 'shared expert combine gate')
        self.router, self.gate_up, self.down = router, gate_up, down
        self.shared_gate, self.shared_up, self.shared_down = shared_gate, shared_up, shared_down
        self.shared_expert_gate = shared_expert_gate
        self.experts, self.dim, self.inter, self.shared = experts, dim, inter, shared
        self.top_k, self.scoring, self.renormalize = top_k, scoring, renormalize
        self.max_output_bytes, self.workspace_bytes = max_output_bytes, workspace_bytes

    def _scratch_bytes(self, tokens):
        # Routing logits/probabilities plus per-slot expert intermediates
        # (gate+up, activated product, down output, weighted contribution)
        # plus the shared expert intermediates and combine gate.
        return 4 * tokens * (2 * self.experts + self.top_k
                            + self.top_k * (3 * self.inter + 2 * self.dim)
                            + 3 * self.shared + self.dim + 1)

    @torch.inference_mode()
    def forward(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.dtype is not torch.float32 or hidden.device.type != 'cpu'
                or hidden.shape[2] != self.dim or not torch.isfinite(hidden).all()):
            raise ValueError('qwen3.5-MoE FFN needs finite CPU FP32 [batch, tokens, hidden] states')
        batch, tokens, _ = hidden.shape
        if not 1 <= batch or not 1 <= tokens:
            raise ValueError('qwen3.5-MoE FFN needs a positive batch and token count')
        flat = hidden.reshape(batch * tokens, self.dim)
        if self._scratch_bytes(batch * tokens) > self.workspace_bytes or 4 * flat.numel() > self.max_output_bytes:
            raise ValueError('qwen3.5-MoE FFN execution budget exceeded')
        coefficients, selected = select_experts(F.linear(flat, self.router), self.top_k,
                                                scoring=self.scoring, renormalize=self.renormalize)
        output = torch.zeros_like(flat)
        for expert in range(self.experts):
            rows, slots = (selected == expert).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            gate, up = F.linear(flat[rows], self.gate_up[expert]).chunk(2, -1)
            contribution = F.linear(F.silu(gate) * up, self.down[expert])
            output.index_add_(0, rows, contribution * coefficients[rows, slots].unsqueeze(-1))
        combine = torch.sigmoid(F.linear(flat, self.shared_expert_gate))
        shared = F.linear(F.silu(F.linear(flat, self.shared_gate)) * F.linear(flat, self.shared_up),
                          self.shared_down)
        output += shared * combine
        if not torch.isfinite(output).all():
            raise ValueError('nonfinite qwen3.5-MoE FFN result')
        return output.reshape(batch, tokens, self.dim)
