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

## Text-path executor: layers 0-3 end to end on real weights

`scripts/experiments/qwen35_moe_text_executor.py` runs the first four
consecutive text layers of the real snapshot end to end — embedding lookup,
three gated-deltanet linear-attention layers, one full-attention layer
(partial rotary 64/256 at theta 1e7, qk-norm, q_proj-packed output gate,
GQA 16/2), our MoE FFN per layer, final norm — against the eager upstream
`Qwen3_5MoeDecoderLayer` stack (Transformers 5.20.2) over the same real
tensors. Ours is the qualified dense-hybrid math (`damendaemon/hybrid.py`
attention and linear-attention methods) ported to the qwen3_5_moe tensor
names, including the two norm conventions the upstream module actually uses:
zero-centered `(1.0 + w)` for layernorms/qk-norms, plain `w` for the
linear-attention gated norm. Bounded slice reads, BF16->FP32, ~19 GiB peak on
a 32 GB host, no staging copy, read-only NAS untouched.

Fixed-seed synthetic random-ID probe [1, 32, 2048] (not real text); evidence
`qwen35-moe-text-executor-layers0-3.json` (71 tensor digests):

- linear layers 0-2 (ours sequential recurrence vs upstream chunked kernel):
  max_abs 2.1e-07 / 2.7e-07 / 7.2e-07, rms_rel up to 1.7e-03
- full-attention layer 3 (op-for-op eager port): max_abs 6.6e-07
- final hidden after the model norm: max_abs 1.7e-05, rms_rel 2.5e-04

The divergence class is exactly sequential-vs-chunked accumulation plus
residual-stream carry-through — no semantic mismatch signature. Claim: our
full text-path implementation matches the upstream eager reference on real
checkpoint weights for these four layers and this probe, at float tolerance.
Not claimed: bitwise equality for the mixer layers, real-activation
behavior, full 40-layer run, GPU, performance, quality.

Next ordered cell: extend to the full 40-layer text path on labnode3 host
memory (bounded per-layer slice streaming), then quantization/placement for
GPU residency.



## Full 40-layer text-path run (streamed, labnode3)

The same executor, per-layer slice-streamed (one layer ~3.5 GiB FP32 resident
at a time; an all-resident FP32 pass would need ~132 GiB), ran all 40 text
layers on labnode3 against the eager upstream stack (Transformers 5.20.2,
isolated --target install, md5-verified transfer). Fixed-seed probe [1, 32,
2048]: all layers agree — per-layer max_abs 1.2e-07 (min) / 1.2e-06 (median)
/ 7.6e-06 (worst, layer 39 full attention); final hidden max_abs 3.0e-05,
rms_rel 2.4e-04. Two linear layers (9, 10) show rms_rel spikes (0.10, 0.015)
with max_abs identical to neighbors — near-zero-output inflation of the
relative metric, not propagation (layer 11 back to 4.5e-04). Peak RSS ~8 GB,
~254% CPU, ~12 min wall. Evidence: qwen35-moe-text-executor-full40.json (692
digests). Claimed: our text path matches the upstream eager reference on real
weights for all 40 layers at float tolerance. Not claimed: mixer bitwise
equality, real activations, generation quality, GPU, performance.
