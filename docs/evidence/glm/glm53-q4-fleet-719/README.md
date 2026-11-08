# GLM-5.3 q4 fleet execution — rung 1: packing + drift record (#719)

2026-11-08, labnode3 (`dsv4f-venv`, 16 threads) against the staged official
checkpoint (NAS original, revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`,
whole-dir model_id `7d4d796b…`). This rung is the weight-level foundation for
the quantized executor ladder on #719: decode the shipped quantization
truthfully, pack with the repo's canonical Q4_0, record paired-delta drift, and
project the full contract to q4 bytes for the fleet plan.

## Source quantization conventions (measured from real tensors)

- GLM-5.3 ships **blockwise-E4M3-FP8** weights with `*_scale_inv` companions
  on a **128×128 element grid with partial tail blocks** (e.g.
  `kv_a_proj_with_mqa` 576×6144 carries a 5×48 scale grid; 576 = 4×128 + 64).
  Truth = `float8_e4m3fn(byte) * scale_inv` expanded over both axes.
- `lm_head.weight` ships **unquantized floating** (no scale companion).
- Attention, indexer, routed-expert and shared-expert weights are all
  blockwise-FP8.

## Weight-level Q4_0 gates — 8/8 PASS

Packing: `damendaemon.quantization.quantize_q4` (uint8 paired nibbles +
FP16 per-32 scales — the repo's canonical format the qualified Triton/CUDA
kernels consume). Reference: the FP32 dequantized truth above (q4-of-FP8,
the honest baseline — not q4-of-BF16).

| tensor | rel RMS | max_abs / absmax | packed ratio | idempotent |
|---|---|---|---|---|
| layers.0 q_a_proj | 0.0908 | 0.0625 | 0.562 | ✓ |
| layers.0 indexer.wk | 0.1012 | 0.0584 | 0.562 | ✓ |
| layers.6 kv_a_proj | 0.0955 | 0.0625 | 0.562 | ✓ |
| layers.6 q_b_proj | 0.1024 | 0.0680 | 0.562 | ✓ |
| experts.0 gate_proj | 0.0886 | 0.0979 | 0.562 | ✓ |
| experts.255 down_proj | 0.0871 | 0.0568 | 0.562 | ✓ |
| shared_experts up_proj | 0.0877 | 0.0625 | 0.562 | ✓ |
| lm_head (bf16 source) | 0.0811 | 0.0621 | 0.281 | ✓ |

- **Paired-delta drift**: 8.1–10.2% rel RMS, worst max_abs ≤ 0.098 of the
  tensor's absmax — the same drift family as the repo's pinned qwen3.8
  FP16→q4_0 record (9.9% RMS mean, docs/evidence/quantization/), here
  measured q4-of-FP8-truth (an extra quantization stage on already-quantized
  weights, honestly disclosed).
- **Idempotency**: re-packing the unpacked values reproduces the packed
  bytes exactly on all 8 (the group-absmax element round-trips by
  construction).
- **Compression**: 0.562× the FP8 source bytes (4-bit + FP16/32 scales over
  8-bit + scales) for packed matrices; 0.281× for the bf16 lm_head.

Packing decisions (mirrored from `quantization.prepare`): all 2-D matrices
packed **except** embeddings, routers (`mlp.gate.*` +
`e_score_correction_bias`) and shared-expert combine gates — selection
semantics stay high precision (pinned exact, #668).

## Full-contract q4 projection (fleet plan input)

At the #56 v5 boundary-legal ranges — stage totals with the same
decisions above:

- Total q4: **395.93 GiB** (vs 698.15 GiB fp8-retained)
- Per stage: labnode3 [0,62) **309.6 GiB** · rocky10 [62,70) **41.4 GiB** ·
  3090box [70,78) **45.0 GiB**

Every stage now fits host RAM with enormous margin (vs 549/74/79 GiB at
fp8); stage-0 at 309.6 GiB even leaves room for cache/workspace on the 589
GiB VM. The 4-host split (adding homer, CUDA→ROCm boundary) is projected
comfortably within this budget.

## Files

- `rung1/report.json` — all rows + projection (the machine-readable record)
- `rung1/rung1-q4.log` — full cell log
- `rung1/glm719_rung1_q4.py` — the cell (blockwise-FP8 decode with partial
  128×128 tail blocks; this decoding logic is the source stage for the q4
  packer on rung 2)

No GPU execution, no end-to-end generation, no quality claim — weight-level
record only. Rung 2: single-group GPU spot cell on the A6000 (packed q4
executor vs the fp8 reference oracle, rung-4 cell-C shape).
