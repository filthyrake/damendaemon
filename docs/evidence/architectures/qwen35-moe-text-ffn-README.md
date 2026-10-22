# Qwen3.5-MoE host text FFN — cell 2 of the #14 Qwen target

Host reference FFN for the pinned qwen3_5_moe checkpoint contract
(`hybrid_moe_checkpoint`, cell 1, merged in `1ecb9243`). This cell implements
and controls the text-path MoE FFN structure that the contract evidences:
router [experts, hidden], fused stacked experts (`gate_up_proj`
[experts, 2*inter, hidden] / `down_proj` [experts, hidden, inter]), dense
shared expert, and the [1, hidden] sigmoid `shared_expert_gate` combine gate.

## What is implemented

`damendaemon/hybrid_moe_ffn.py` — `HybridMoeFeedForward`:
- FP32 CPU arithmetic only; weights borrowed immutably.
- Routing reuses the audited `damendaemon.expert_selection.select_experts`
  (scoring policies softmax/sigmoid, optional top-k renormalization with the
  documented underflow guard, lower-ID tie contract).
- Per-expert gathered forward (one pass per active expert, `index_add_`
  accumulation), matching the upstream qwen3_moe loop structure
  (Transformers 4.56.1, `Qwen3MoeSparseMoeBlock`).
- Gated shared expert: `shared(x) * sigmoid(x @ shared_expert_gate^T)`.
- Explicit workspace (routing + per-slot intermediates) and output byte
  budgets with loud over-budget rejection.

## Evidence

`tests/test_hybrid_moe_ffn.py` (torch CI job, `runtime-integration`):
- **Independent control**: a per-token naive reference with plain torch ops
  (no shared code with the implementation) matches `forward` across all four
  policy combinations — sigmoid/softmax scoring with and without top-k
  renormalization (atol/rtol 1e-5; different accumulation order).
- Batch/token shape sweep, finite-output check.
- Rejection surface: policy (top_k bounds, scoring, renormalize, budgets),
  geometry (every tensor family, rank, dtype, non-finite), inputs (dtype,
  empty batch, rank), workspace and output budget overflows.

Local: 8/8 pass on torch 2.14.1 (macOS). CI runs the same suite on the pinned
torch 2.6.0 CPU build (ubuntu-24.04).

## Explicitly not claimed

- **Routing policy is now pinned** to the eager reference implementation
  (Transformers 5.20.2 `Qwen3_5MoeTopKRouter`): FP32 softmax over router
  logits, top-k on the probabilities, **unconditional** renormalization of
  the selected weights (no config gate — unlike earlier qwen3_moe, whose
  `norm_topk_prob` defaulted off). The qualified default of
  `HybridMoeFeedForward` encodes this (`scoring='softmax'`,
  `renormalize=True`); sigmoid remains an explicit unqualified option.
- No full-model execution, no GPU, no performance, no quality claims.
- MTP and vision tensors remain excluded per the contract.

## Executor: real-weight single-layer run

`scripts/experiments/qwen35_moe_layer_executor.py` reads one layer's seven
MoE FFN tensors from the real Qwen3.6-35B-A3B snapshot by bounded safetensors
slice reads (BF16->FP32, ~3.5 GiB peak host memory, no full materialization,
no writes to the read-only NAS) and runs three computations over the same
real tensors: our FFN (qualified policy), an independent per-token naive
control, and the eager upstream `Qwen3_5MoeSparseMoeBlock` (Transformers
5.20.2, `_experts_implementation='eager'`).

Layer 3 (full-attention), fixed-seed synthetic normal probe [2, 6, 2048] —
**not real activations**; evidence
`qwen35-moe-executor-layer3.json`:

- ours vs upstream eager reference: **max_abs 0.0, rms_rel 0.0 — bitwise
  identical** (same op sequence and accumulation order over the same tensors).
- ours vs independent control: max_abs 3.6e-08, rms_rel 3.4e-04 (different
  accumulation order; near-zero output elements inflate the relative RMS).
- Per-tensor BF16 digests, config digest, selected experts and weight range
  recorded; real router selected experts [134, 179, 174, 76, 129, 109, 188, 48]
  for the first probe row with renormalized weights spanning 0.10–0.17.

Claim: our FFN implements the pinned upstream semantics exactly on real
checkpoint weights for this layer/probe. Not claimed: full-model execution,
real-activation behavior, GPU, performance, quality.

Next ordered cell: multi-layer/full-text execution path (attention + MoE FFN
against the qualified dense-hybrid machinery), then quantization/placement
decisions for GPU residency.

