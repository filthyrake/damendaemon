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

## Expert-residency FP4 error budget (cell 6)

`scripts/experiments/qwen35_moe_expert_fp4.py` measures the real-weight error
budget for quantizing the routed-expert bank (router, shared expert and
combine gate stay FP32 — the residency plan), using the audited
`activation_simulation` component simulations (DeepSeek-V4-Flash
conventions: E2M1 nearest/even with one E8M0 power-of-two scale per 32
values; E4M3FN with one scale per 64) on real layer tensors across depth
(layers 0, 3, 9, 20, 39), fixed-seed synthetic probe [2, 6, 2048].
Simulated quantization substitutes dequantized FP32 tensors into the audited
FFN, so the FP32 op order matches the cell-3 reference exactly and the
deviation is purely the quantization effect. Evidence:
`qwen35-moe-expert-fp4-budget.json` (35 digests).

- fp8 (E4M3FN): max_abs 2.4e-04 (layer 0) .. 1.4e-03 (layer 39, worst).
- fp4 (E2M1+E8M0): 1.1e-03 .. 5.3e-03 (worst, layer 39); typical ~2e-03.
- fp4 with normalized-Hadamard weight rotation (effective dequant(W@H)@H):
  9.9e-04 .. 4.9e-03 — a wash vs plain fp4 on these weights (better on 3/5
  layers, worse on 2), so rotation buys nothing decisive here.

rms_rel values are near-zero-output-inflated (established in the 40-layer
run); max_abs is the load-bearing metric. Bank memory arithmetic (computed,
not measured): 60.0 GiB BF16 / 30.5 GiB E4M3FN / 15.9 GiB E2M1+E8M0 for the
full 40-layer bank — E2M1 is the only single-GPU option on the 24-GiB A30
(E4M3FN does not fit; BF16 needs none of the GPUs).

Claimed: per-layer FFN-output error budget of E2M1/E8M0 and E4M3FN
weights-only expert quantization on real weights, sampled across depth.
Not claimed: error compounding through 40 layers, logits/generation impact,
quality acceptability, GPU execution. Whether 5e-03-class per-layer FFN
deviation is acceptable for serving quality is a downstream qualification
gating decision, not a measurement made here.

## E2M1 bank residency on the A30 (cell 7)

`scripts/experiments/qwen35_moe_a30_residency.py` places the full 40-layer
routed-expert bank on the A30 under the shared-CUDA guard (UUID-pinned, TF32
off, phase completed) and composes layers 3 and 39 through the prepared CUDA
MoE stack. Packing is transcribed from `simulate_fp4` (same boundaries,
nearest-even ties, power-of-two scales per 32 values) into the prepared
e2m1 layout, runs on-GPU in 32-expert chunks, and is asserted exact:
`decode_tile(pack(w)) == simulate_fp4(w)` on a random sample, a real gate_up
tile and a GPU-packed slice. Measured: 15.95 GiB allocated vs 15.94 GiB
arithmetic (18.54 reserved). Composition vs the cell-6 CPU control
(pinned-quantization, shared expert zeroed): max_abs 3.260e-09 (layer 3) /
1.118e-08 (layer 39) — the GPU reproduces the pinned E2M1 semantics to
op-order noise (fused coefficients fold pre-down vs the control's post-down;
same math). Vs the FP32 reference: 1.877e-03 / 5.268e-03 — exactly the
cell-6 budget; residency adds nothing to the quantization error. Routes
asserted equal. Evidence: `qwen35-moe-a30-e2m1-residency.json` + guard
report. Claimed: bank residency + composition fidelity. Not claimed:
throughput, full-model GPU execution, quality.

## Single-GPU 40-layer forward and the E2M1 compounding finding (cell 8)

`scripts/experiments/qwen35_moe_a30_full_forward.py` runs all 40 text layers
in one guarded A30 process: E2M1 bank resident (15.95 GiB, cacheable),
non-expert FP32 weights streamed per layer (full bank+weights residency
~21.7 GiB exceeds the guard envelope - the reserve gate correctly refused
the first all-resident attempt at layer 30), fused composition chunk-256,
peak 16.09 GiB, phase completed, TF32 off. The first streaming attempt
honestly failed the 1800s guard deadline (~85 s/layer); the packed-bank cache
plus chunk-256 (~25 s/layer) brought it inside one window.

GPU (E2M1 bank) vs host FP32 reference, max_abs by depth: 6.3e-03 (layer 0),
3.3e-02, 1.1e-01, 1.9e-01, 1.65 (layer 39), 11.27 final after the norm.
Host final max 57.2 / RMS 2.24: the deviation reaches ~20% of peak, ~5x RMS.
The cell-6 per-layer budget (~2e-03) compounds ~5000x - super-linear, not
the ~6x of quadrature - consistent with route flips amplifying through the
residual stream. rms_rel 55.8 is near-zero-inflated; max_abs is load-bearing.

**Negative result, recorded exactly:** E2M1 weights-only quantization of the
entire routed bank does not survive 40-layer depth at usable precision.
Hadamard rotation was already a wash (cell 6). Consequence: the precision
path is E4M3FN experts (1.4e-03/layer, 30.5 GiB bank - does not fit one
A30), so the next cell is bank split across A30+A2 or per-layer E4M3
host streaming. No generation-quality cell should run on the E2M1 bank.

## E4M3 compounding measurement (cell 9)

Same single-GPU forward, expert bank E4M3FN (per-128x128-block E8M0 scales,
DeepSeek-V4-Flash convention; packing exactness asserted against an
independent CPU transcription and the prepared decoder) packed per layer
from streamed FP32 - the 30.5-GiB bank is never resident; peak GPU 0.9 GiB,
guard completed. Probe/host reference identical to cell 8 (seed 20261023).

E4M3 vs E2M1 max_abs by depth (both vs the same host FP32 reference):
layer 0: 1.2e-03 vs 6.3e-03 (5.1x) ... layer 39: 1.27 vs 1.65 (1.3x);
final: 3.49 vs 11.27 (3.2x). The format gap collapses with depth - both
quantization formats land in the same O(1)-deviation basin (~1.6x host RMS,
6% of peak for E4M3): a trajectory-decorrelation signature. Compounding is
again super-linear (per-layer 1.2e-03 -> ~2900x at final; quadrature ~6x).

**Conclusion, recorded exactly:** super-linear compounding through route
flips is structural to weights-only expert quantization at either format,
not a precision class that a better format escapes. Hidden-state fidelity
through 40 layers requires unquantized experts (host-streamed BF16, the
only exact path; ~1.5 GiB/layer transfers). The standard acceptance
methodology for quantized deployments - generation-quality gates on real
text (perplexity, greedy agreement, logit divergence) - is the arbiter for
whether the E4M3 bank is deployable; hidden-state fidelity is not that
gate. The epic's next decision: E4M3 quality-gated generation cell vs
exact-but-slower BF16 per-layer streaming. Evidence:
qwen35-moe-a30-e4m3-compounding.json + guard report.

## Generation-quality gate (cell 10)

The arbiter cell 9 deferred to. Harness
`scripts/experiments/qwen35_moe_a30_generation_quality.py`: four fixed
model-card text prompts tokenized with the snapshot tokenizer (4 x 46
tokens), full 40-layer forward to lm_head logits, reference = host FP32
ported math (cell 5, verified vs eager upstream at rms_rel 2.4e-04),
candidate = the cell-9 E4M3 per-layer streamed bank under
`guard_shared_cuda` on the A30 (one fused route chunk per layer; peak
0.91 GiB; guard completed in 1409 s).

Gates pre-declared before any run, thresholds unchanged at evaluation:

- greedy top-1 agreement >= 0.80 over positions >= 8 (warmup removed)
- mean KL(ref || cand) <= 1.0 nat over the full 151936-vocab
- candidate/reference perplexity ratio <= 1.10

Measured (as run, no relaxation): agreement **0.9932**, mean KL
**0.00424 nat**, perplexity ratio **1.0081** (NLL 2.588 vs 2.580). All
three gates PASS. Divergence canaries (added in review remediation, run
recorded by it): logit_max_abs **1.66** - the candidate is provably not
the reference - and all-positions agreement 0.9946. Reproducibility:
two independent guarded runs of the candidate agree to GPU-reduction
noise (agreement bit-identical; KL/NLL within ~1e-9). Scope, recorded
exactly: three pre-declared metrics on four fixed model-card prompts for
this snapshot at <= 64 tokens; no throughput, long-context, or
other-prompt-set claims. Evidence:
qwen35-moe-generation-quality-e4m3.json + guard report; reference
manifest (ids digest + shape) in the run log.

**Cell-10 conclusion:** the E4M3 per-layer-streamed expert bank is
deployable in kind - at this prompt scale it is numerically
indistinguishable from the FP32 reference on greedy continuation,
next-token distribution, and perplexity, despite the cell-9 hidden-state
decorrelation (final max_abs 3.49). Route-flip compounding degrades
hidden-state fidelity but not, at this scale, generation quality. The
structural cell-8/9 finding stands; the deployability question it raised
is answered in kind, at the pre-declared gates.
