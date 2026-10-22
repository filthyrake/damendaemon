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

- **Routing policy is not qualified.** The qwen3_5_moe_text config pins no
  routing keys (no `norm_topk_prob`, no scoring field). Scoring and
  renormalization are explicit constructor policies here. Upstream qwen3_moe
  uses softmax with config-gated renormalization (Transformers 4.56.1 default
  off); whether qwen3_5_moe_text uses sigmoid scoring (like GLM) or softmax
  (like qwen3_moe) is unresolved and must be pinned against the real
  checkpoint in the executor cell — the same discipline that caught the
  q_proj gate packing.
- No real-checkpoint execution, no GPU, no performance, no quality claims.
- MTP and vision tensors remain excluded per the contract.

Next ordered cell: real-checkpoint staging decision (67 GiB / 26 BF16 shards)
and the first executor run against pinned weights.
