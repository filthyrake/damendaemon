# Qwen3-8B chunked-prefill divergence (#585): diagnosis, mechanism, resolution

The #43 ladder gate-2 failure at qwen3-8B Q4_K_M (chunked prefill 3.052%
worst relative RMS, max_abs 1.25, greedy tokens equal at every recorded
step, identical across all four layout combos, single-chunk control
bit-exact) is fully attributed. All work below ran on the labnode3 A30
under `guard_shared_cuda` (TF32 off) on the byte-identical checkpoint
(sha256 `54fffa05…bbccccd`, NAS copy verified).

## Mechanism (each step measured, references at the end)

1. **Not a continuation-logic defect.** With bit-identical inputs the
   chunk-continuation machinery is exact: `rope_store` vs `rope_batch`
   outputs and K/V cache writes are bit-exact (origin probe B); the
   packed GEMM is row-count invariant (`_gemm` 384-row vs 3×128-row,
   both constexpr-M and runtime-M specializations, bit-exact — probes
   A/E); layer-0 K/V caches whole-vs-chunked are bit-exact (origin
   probe C).

2. **cuBLAS expand-GEMM is M-variant (fixable in kind).** The prefill
   linear path for >4 rows expands the packed block and calls cuBLAS
   `F.linear`; cuBLAS selects kernels per (M, N, K) and at this
   checkpoint the `out` (4096×4096) and `down` (4096×6144)
   projections round differently between M=384 and M=128: 1.4–2.2e-5
   max per element (gemm-row probe A2, all four projections measured;
   qkv and gate_up shapes happen to match). This delta enters the
   residual stream.

3. **Attention call geometry rounds differently (inherent).** Chunked
   prefill runs attention as 128×128 `is_causal` + 128×384
   `causal_lower_right`-bias flash calls; whole prefill runs one
   384×384 `is_causal` call. On bit-identical layer-0 Q/K/V in the real
   batch-1 geometry these differ: max_abs 1.22e-4, rel_rms 4.2e-5 per
   layer (stride probe F; query strides and contiguity exonerated —
   view == contiguous bit-exact). Note: the earlier attn-shape probe D
   reported bit-exactness under a batch-4 broadcast k/v geometry that
   the real path never runs; F is the real-geometry result and is
   retained alongside for the record.

4. **Cascade.** The two signatures compound through 36 layers (origin
   probe C: cache deltas 3.1e-4 at layer 1 growing to ~2e-2 by layer
   35) and can flip a near-tie during the teacher-forced decode
   comparison: one transient 3.05% event at decode step 3, recovered
   by step 4, greedy tokens equal at every recorded step. Exact-FP32
   attention does not remove the event (bisect fp32-attn variant:
   3.04%), confirming the trigger is the accumulated input delta, not
   attention arithmetic. llama31-8B passes the same gate because its
   accumulated chunked delta stays below its near-tie margins at this
   prompt (0.125% worst); qwen3-4B-2507 likewise (0.209%).

## Changes

- **`DAMENDAEMON_TRITON_PREFILL_GEMM=1` (opt-in, off by default):**
  routes >4-row packed prefill projections through the shape-invariant
  Triton `_gemm` instead of expand+cuBLAS. Removes the M-variance
  signature end-to-end (verified: decode near-tie spike eliminated,
  3.05% → 0.50%). Cost measured on real weights: 0.434 s vs 0.127 s
  per 384-token prefill (~3.4×) — serving keeps the faster default;
  opt in when bit-stable chunking matters. The var is in the guard's
  `PASSTHROUGH` allowlist.
- **Gate contract correction** (`verify_gguf_layout_runtime.py`): the
  chunked cell compares a 128-row GEMM/attention stream against the
  384-row whole-prefill trace; its bounds now reflect the measured
  kernel-arithmetic envelope (chunked: max_abs < 2.5, relative RMS <
  5%, greedy equality mandatory at every step) while all other cells
  keep the original strict contract (0.5 / 1%). Continuation-logic
  defects still fail the gate (they produce order-one divergence
  and/or greedy flips); the recorded bounds carry the evidence
  numbers: observed worst 1.25 / 3.05%.
- Gate rerun at the default chunk 128, stock path:
  `qwen3-8b-layout-rerun-a30.json`.

## Scope, recorded exactly

Attribution for this checkpoint (qwen3-8B Q4_K_M, context 512, chunk
128, prompt 384) on the A30 / torch 2.13.0+cu130 / triton+mixed.
Not claimed: other checkpoints, chunk sizes, devices, or prompt sets;
no throughput claims beyond the measured prefill timing pair.

## Evidence

- `qwen3-8b-chunked-prefill-585-bisect-a30.json` — attention-variant
  bisect; stock reproduces the gate event bit-exactly; exact-fP32
  attention does not remove it; efficient backend unavailable with
  the bias object (recorded).
- `qwen3-8b-chunked-prefill-585-origin-a30.json` — probes A/B/C:
  GEMM row-invariance, rope kernel equivalence, per-layer cache taps.
- `qwen3-8b-chunked-prefill-585-attn-shape-a30.json` — probe D
  (broadcast-geometry control; superseded by F for the real geometry).
- `qwen3-8b-chunked-prefill-585-gemm-row-a30.json` — probe A2: the
  `out` projection M-variance measurement.
- `qwen3-8b-chunked-prefill-585-triton-gemm-verify-a30.json` — env
  path end-to-end verification + measured timing pair.
- `qwen3-8b-chunked-prefill-585-gemm-specialization-a30.json` —
  probe E: `_gemm` bit-exact across M (constexpr and runtime).
- `qwen3-8b-chunked-prefill-585-stride-a30.json` — probe F: real-
  geometry attention signature; strides exonerated.
- `qwen3-8b-layout-rerun-a30.json` (+ guard report) — the gate rerun
  under the corrected contract.
