# GLM-5.3 q4 fleet execution — rung 3a: HOMER 8060S (gfx1151) parity (#719)

2026-11-08, homer (Ryzen AI Max+ 395, Radeon 8060S, ROCm 7.13 / torch
2.11.0+rocm, HIP 7.13.26162), guarded (`guard_homer`, 80 °C limit, 900 s)
plus the in-cell fail-closed `CoolingGuard`. Standing permission applied:
sheena stack stopped cleanly before the cell (chat/detect/embed/rerank
listeners gone, VRAM freed), restored and verified after (all listeners
back, fans auto, strix-fan-daemon active).

## Result — ALL GATES PASS (attempt 2, `evidence2.json` / `guard2.json`)

Paired-oracle design identical to rung 2, scoped for the APU: 48 real
layer-6 routed experts (of 256) + the full shared expert, real router-row +
correction-bias subset, both banks resident through the production
`CudaPackedGlmFeedForward`. Oracle = shipped blockwise-E4M3-FP8 storage
placed directly; candidate = on-device canonical Q4_0 packing. **Pure-torch
packed execution on HIP — PyTorch's ROCm build presents devices through the
CUDA API, so the qualified CudaPacked* machinery ran unmodified; no native
NVIDIA libraries anywhere on the path** (rocm_policy refuses those; this
path never touches them).

1. **Per-expert parity**: 32 routed experts selected on the pinned inputs +
   shared — worst rel RMS **0.1494** ≤ 0.15 ✅
2. **Full FFN parity**: rel RMS **0.1239**, max_abs 0.087 × absmax ✅
3. **q4 composition equivalence**: batched vs per-token **bit-identical
   (0.0)** ✅
4. **Memory**: q4 bank **1.040 GB** vs fp8 1.849 GB (0.562×); peak reserved
   3.14 GB; peak Tctl 38 °C (edge 30 °C), fans fixed/5 throughout
   (4219/4312/1818 RPM readback).

No token, quality, throughput or SLO claim. Bring-up capability + parity
record only.

## Attempt 1 retained (`evidence.json` / `guard.json`)

Refused at key-check in 1.2 s: a prefix-construction bug in the runner
dropped the `experts.` path segment (`mlp.0.gate_proj` — not in the index).
No GPU time spent; the fail-closed surface worked (CoolingGuard verified
fans/sensors before the refusal).

## Significance for the lane

The AMD half of the CUDA→ROCm fleet boundary is proven at the packed-executor
level: same canonical q4 layout, same production FFN, same real weights,
drift statistically indistinguishable from the A6000's (worst 0.1494 vs
0.1464; FFN 0.1239 vs 0.1197) — gfx1151's rounding shows no extra divergence
beyond the shared q4-of-FP8 family envelope. Rung 3b (the 4-host chain) can
place homer's tail stage on this qualified surface.
