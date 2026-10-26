"""TokenGate: learned prefill token pruning for the hybrid executor (#641).

Implements the published TokenGate mechanism (arXiv:2610.26754) natively
against the damendaemon hybrid executor surfaces, from the Apache-2.0
upstream reference (token-gate-llms/TokenGate, pinned 6e51aac in the #640
evaluation). No new runtime dependency: the gate is plain torch math.

Mechanism (mirroring upstream semantics exactly):

- A scoring branch — a copy of the gate layer's MLP whose ``down_proj`` is
  replaced by a zero-initialized (1, intermediate) readout row — scores each
  PRUNABLE token's layer-L phase boundary ``a_j`` (the post-attention
  residual before the layer's MLP half): ``score_j = W_read · SwiGLU(LN2(a_j))``.
- The E2EGate turns scores into a per-(layer, head) key bias:
  ``b_j[l,h] = softplus(gate_scale[l,h]) * RMSNorm_seq(score)_j`` — the RMS
  normalization runs over the ARTICLE (prunable) axis with a frozen gain.
- Selection: the aggregate (first bias layer's slice, mean over heads)
  top-K over prunable tokens; non-prunable tokens (everything outside the
  declared article span) always survive. Layers above L process only the
  kept tokens; full-attention layers above L write KV only at kept
  positions, and their logits get
  ``(r_q · softplus(attn_scale[l,h])) · (q·k/√d) + b_k`` — the per-query
  temperature r_q = log(m_q)/log(p_q) (kept slot vs original position,
  clamped ≤ 1; == 1 when nothing is pruned, so an untouched gate at init
  reproduces the base model). DeltaNet layers above L carry no per-token
  KV and receive no bias; they simply run on the kept sequence.
- Safety semantics (damendaemon addition, tested): an all-zero score vector
  means the gate is untrained/inactive — the executor falls back to the
  standard prefill path, byte-exactly, at any configured K.

Decode keeps upstream's fold: the hole mask excludes never-written KV
positions, the query uses its own s_eff, and the per-key bias enters as
the head-mean folded additive term.
"""
import math
import os

import torch
from torch.nn import functional as F

# Upstream _RMS_EPS: matches the Qwen3 family rms_norm_eps used by the gate's
# sequence normalization (e2e_gate_module.py, pinned 6e51aac).
RMS_EPS = 1e-6


def _softplus_inverse(value):
    """raw parameter whose softplus() == value (upstream init convention)."""
    return math.log(math.expm1(value))


class TokenGate:
    """Gate weights + selection/bias math, executor-agnostic.

    ``bias_layers`` holds the FULL-ATTENTION layer indices above the gate
    layer, in order — row r of gate_scale/attn_scale belongs to
    bias_layers[r] (upstream n_layers_bias ordering).
    """

    def __init__(self, config, gate_layer, bias_layers, heads, keep):
        if (type(gate_layer) is not int
                or not 0 <= gate_layer < config['num_hidden_layers']
                or not isinstance(bias_layers, list)
                or any(config['layer_types'][i] != 'full_attention' for i in bias_layers)
                or any(i <= gate_layer for i in bias_layers)
                or type(heads) is not int or heads <= 0
                or type(keep) is not int or keep < 0):
            raise ValueError('invalid token gate configuration')
        self.gate_layer = gate_layer
        self.bias_layers = bias_layers
        self.heads = heads
        self.keep = keep
        self.rows = len(bias_layers)
        self.hidden = config['hidden_size']
        self.intermediate = config['intermediate_size']
        # Branch: SwiGLU MLP with the absorbed (1, intermediate) readout row.
        self.branch_gate = torch.zeros(self.intermediate, self.hidden)
        self.branch_up = torch.zeros(self.intermediate, self.hidden)
        self.branch_down = torch.zeros(1, self.intermediate)
        # Effective magnitude softplus(beta) per (bias row, head). Untrained
        # raw zero is never interpreted as a trained value: zero scores keep
        # the gate inactive regardless (see select()).
        self.gate_scale = torch.zeros(self.rows, self.heads)
        # Temperature softplus(sigma); upstream init softplus^-1(1) so the
        # untrained effective temperature is exactly 1.
        self.attn_scale = torch.full((self.rows, self.heads), _softplus_inverse(1.0))
        self.norm_gain = torch.ones(1)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def untrained(cls, executor, gate_layer, keep):
        """Upstream's step-0 initialization from a live executor.

        The branch's gate/up matrices clone the gate layer's own MLP
        (upstream clones the trained block MLP so the branch starts as the
        layer's tied readout); the readout row is zero-init, so every score
        is exactly zero and the gate is born inactive.
        """
        config = executor.c
        heads = config['num_attention_heads']
        bias_layers = [i for i in range(gate_layer + 1, executor.layer_end)
                       if config['layer_types'][i] == 'full_attention']
        if not bias_layers:
            raise ValueError('token gate needs a full-attention layer above the gate layer')
        gate = cls(config, gate_layer, bias_layers, heads, keep)
        w = executor.w
        gate.branch_gate = w[f'{gate_layer}.gate_up'][:gate.intermediate].clone().float()
        gate.branch_up = w[f'{gate_layer}.gate_up'][gate.intermediate:].clone().float()
        # Zero readout row: scores are all zero at init (upstream absorbed
        # readout zero-init). gate_scale stays at the raw zero: the gate is
        # inactive until scores are nonzero, and trained checkpoints carry
        # their own scales.
        return gate

    @classmethod
    def from_upstream(cls, path, config, gate_layer, keep):
        """Load an upstream E2EGate checkpoint (gate_step*.pt, #640 pins).

        weights_only=True: the upstream save format is a plain dict of
        tensors plus a metadata dict of strings/ints (verified against the
        released liana313 checkpoints in the #641 drift cell); no pickled
        objects are expected, so the restricted loader is both safe and
        sufficient.
        """
        blob = torch.load(path, map_location='cpu', weights_only=True)
        state, metadata = blob['state_dict'], blob['metadata']
        if int(metadata['gate_layer']) != gate_layer:
            raise ValueError(f"gate checkpoint layer {metadata['gate_layer']} != {gate_layer}")
        heads = int(metadata.get('n_heads', config['num_attention_heads']))
        bias_layers = [i for i in range(gate_layer + 1, config['num_hidden_layers'])
                       if config['layer_types'][i] == 'full_attention']
        gate = cls(config, gate_layer, bias_layers, heads, keep)
        gate.gate_scale = state['gate_scale'].clone().float()
        gate.attn_scale = state['attn_scale'].clone().float()
        gate.norm_gain = state['b_norm.weight'].clone().float().reshape(1)
        branch = {k[len('gate_branch.'):]: v for k, v in state.items()
                  if k.startswith('gate_branch.')}
        gate.branch_gate = branch['gate_proj.weight'].clone().float()
        gate.branch_up = branch['up_proj.weight'].clone().float()
        gate.branch_down = branch['down_proj.weight'].clone().float()
        if (tuple(gate.gate_scale.shape) != (gate.rows, gate.heads)
                or tuple(gate.branch_down.shape) != (1, config['intermediate_size'])
                or tuple(gate.branch_gate.shape) != (config['intermediate_size'], config['hidden_size'])):
            raise ValueError('gate checkpoint geometry does not match the model configuration')
        return gate

    # ── gate math ──────────────────────────────────────────────────────────

    def _branch(self, z):
        gate = F.linear(z, self.branch_gate)
        up = F.linear(z, self.branch_up)
        return F.linear(F.silu(gate) * up, self.branch_down).squeeze(-1)

    def score_tokens(self, boundary, norm2_weight, eps):
        """Raw per-token scores from the layer-L phase boundary.

        ``boundary`` is a_j (post-attention residual of layer L, before its
        MLP half); ``norm2_weight`` is the layer's post_attention_layernorm
        weight in the executor convention (loaded weight + 1); ``eps`` is
        the model's rms_norm_eps. Mirrors upstream gate_branch_hs with the
        absorbed readout.
        """
        z = boundary.float()
        variance = z.square().mean(-1, keepdim=True)
        z = z * torch.rsqrt(variance + eps) * norm2_weight.float()
        return self._branch(z)

    def bias(self, scores, article_mask):
        """(n, length, rows, heads) key bias; zero on non-article tokens.

        RMS-normalize over the ARTICLE axis with the frozen gain, then
        scale by softplus(gate_scale). The same normalized pattern fills
        every (l, h) slice — what differentiates them is gate_scale
        (upstream shared-readout expand).
        """
        n, length = scores.shape
        flat = scores.reshape(-1)
        article_flat = article_mask.reshape(-1) if article_mask is not None else None
        selected = flat[article_flat] if article_flat is not None else flat
        normalized = selected * torch.rsqrt(selected.square().mean() + RMS_EPS) * self.norm_gain
        scaled = F.softplus(self.gate_scale) * normalized.unsqueeze(-1).unsqueeze(-1)
        bias = torch.zeros(n, length, self.rows, self.heads,
                           dtype=scores.dtype, device=scores.device)
        if article_flat is not None:
            bias = bias.reshape(-1, self.rows, self.heads)
            bias[article_flat] = scaled.to(bias.dtype)
            return bias.reshape(n, length, self.rows, self.heads)
        return scaled.reshape(n, length, self.rows, self.heads)

    def aggregate(self, bias):
        """Selection score: first bias row, mean over heads (upstream)."""
        return bias[..., 0, :].mean(-1)

    def select(self, scores, article_mask, keep):
        """Kept article positions by aggregate top-K (upstream selection).

        Non-article tokens always survive (handled by the executor, which
        owns the full kept set); an all-zero article score vector keeps
        every article token — the tested untrained-gate safety semantics
        (byte-exact base model at any configured K).
        """
        device = scores.device
        positions = torch.arange(scores.shape[1], device=device)
        if article_mask is None:
            article_positions = positions
            flat = scores.reshape(-1)
        else:
            article_positions = positions[article_mask]
            flat = scores.reshape(-1)[article_mask.reshape(-1)]
        if keep >= flat.numel() or not flat.any():
            return article_positions
        top = torch.topk(flat, keep).indices
        return article_positions[top.sort().values]

    @staticmethod
    def kept_ratio(kept_positions):
        """r_q = log(m)/log(p): kept slot (1-based) vs original position.

        Clamped so r == 1 for the earliest tokens and r <= 1 always; == 1
        for every token when nothing was pruned (m == p), which is what
        makes an untouched gate reproduce the base model exactly.
        """
        m = torch.arange(1, kept_positions.numel() + 1,
                         device=kept_positions.device, dtype=torch.float32).clamp(min=2.0)
        p = (kept_positions.float() + 1.0).clamp(min=2.0)
        return (torch.log(m) / torch.log(p)).clamp(max=1.0)

    def to_device(self, device):
        self.branch_gate = self.branch_gate.to(device)
        self.branch_up = self.branch_up.to(device)
        self.branch_down = self.branch_down.to(device)
        self.gate_scale = self.gate_scale.to(device)
        self.attn_scale = self.attn_scale.to(device)
        self.norm_gain = self.norm_gain.to(device)

    def temperature(self, row):
        """Effective per-(row, head) softmax temperature softplus(sigma)."""
        return F.softplus(self.attn_scale[row])

    def decode_fold(self, bias, kept_positions):
        """Per-key folded additive term for decode (upstream's head-mean).

        The phase-2 key j contributes mean_h s_eff(r_j) * b_j[l,h] — the
        key's own ratio r_j times the head-mean bias — constant through
        decode, so it is folded into the additive mask once per prefill.
        """
        ratio = TokenGate.kept_ratio(kept_positions)
        return (bias * ratio.view(-1, 1, 1)).mean(-1)


def enabled():
    return os.environ.get('DAMENDAEMON_TOKEN_GATE', '0') == '1'


def configured_layer():
    layer = os.environ.get('DAMENDAEMON_TOKEN_GATE_LAYER', '1')
    try:
        return int(layer)
    except ValueError:
        raise ValueError('DAMENDAEMON_TOKEN_GATE_LAYER must be an integer')


def configured_keep():
    keep = os.environ.get('DAMENDAEMON_TOKEN_GATE_K', '512')
    try:
        value = int(keep)
    except ValueError:
        raise ValueError('DAMENDAEMON_TOKEN_GATE_K must be an integer')
    if value < 0:
        raise ValueError('DAMENDAEMON_TOKEN_GATE_K must be non-negative')
    return value
