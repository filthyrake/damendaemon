# HOMER prefill GEMM precision gap (#645 gate 1)

Date: 2026-10-26 · Host: HOMER (Radeon 8060S, gfx1151) · Stack: torch
2.11.0a0+rocm7.11.0a20260106 + triton 3.6.0 (the production venv) · Runner:
`scripts/experiments/prefill_precision_gap_645.py` (this directory's
`prefill-gap.json`; run log + the retained first-attempt log on HOMER at
`~/npu645-evidence/`)

Thermal: fans fixed@5 for the run (restored after; strix-fan-daemon
restarted), umr package telemetry sampled at 2 Hz throughout
(`telemetry.log`) — package stayed ≤40.5 °C, socket ≤32.7 W peak. A GEMM
microbench is far inside the #68 prefill envelope (74.0 °C peak at 4K).

## What was measured

Four arms at decoder prefill shapes (synthetic weights; kernel-arithmetic
rate, not checkpoint quality — the executor admission scope in
`hybrid_precision.select` is unchanged):

- **q8_compensated / q4_compensated**: the production batch-prefill kernels
  `damendaemon.compensated_q8.linear` / `linear_q4` (packed weights + fp16
  block scales, three BF16 dot products per K-block, FP32 in/out) — the
  Q4 arm is the arithmetic the HOMER 32K path runs.
- **dense_f32 / dense_bf16**: `torch.nn.functional.linear` with FP32 / BF16
  weights (rocBLAS).

Each quantized arm's output was checked against an FP32 dequantized-weight
reference on the host (sanity, not a gate): max_abs 6.4e-4 (Q8, 0.8B
shapes) / 5.7e-3 (Q8, 27B shapes) / ≤4e-4 (Q4 everywhere) — the
compensation working at its designed precision. The first attempt
(`run.attempt1.log`) failed this check for Q4 — the runner's reference
dequant had the nibble order interleaved; fixed (16 lows then 16 highs per
32-group, matching the kernel's byte layout) and re-run. No recorded
numbers were edited; the failed attempt is retained.

## Results (TOPS = 2·M·K·N / t)

| Shape (M, K, N) | q8 comp | q4 comp | dense f32 | dense bf16 |
|---|---|---|---|---|
| 0.8B chunk127 (127, 1024, 3584) | 3.36 | 3.05 | 0.91 | 14.6 |
| 0.8B prefill512 (512, 1024, 3584) | 3.51 | 3.38 | 1.29 | 22.5 |
| 27B chunk127 (127, 5120, 17408) | 3.34 | 3.33 | 0.37 | 7.7 |
| 27B prefill512 (512, 5120, 17408) | 3.16 | 3.18 | 0.36 | 23.0 |

## Gate-1 verdict: the realizable lever is ~2.3–7.3× by shape, not 20–80×

1. **The #644 dense-vs-dense microbench overstated the opportunity.** The
   production compensated Triton kernel already runs **~8.8–9.1×** dense
   FP32 at the 27B shapes (3.16 vs 0.36 TOPS at M=512; 3.3 vs 0.37 at
   M=127) — its three-BF16-dot compensation keeps the inner math at
   BF16-class throughput. The headline "bf16 runs 20–80× f32" compared
   dense arms; the production path is not dense f32.
2. **The remaining ceiling vs the production path is strongly
   shape-dependent**: dense bf16 beats the compensated kernel by
   **~2.3× at the production chunk row count (M=127, 27B shapes: 7.7 vs
   3.3 TOPS)** — dense bf16 loses tile efficiency at M=127 while the
   compensated kernel is flat across M — and by **~6.7–7.3× at M=512**
   (22.5 vs 3.4 at 0.8B; 23.0 vs 3.2 at 27B). A bf16 prefill policy's
   payoff therefore also depends on the *chunk-size policy*: at the
   current 127-row chunk the 27B ceiling is ~2.3×; larger chunks widen
   it toward ~7× at the cost of prefill-peak memory.
3. **Why the compensated kernel runs 2.3–7.3× off the dense bf16 rate**
   (it already does BF16 dots): it pays for (a) on-the-fly dequantization
   (packed loads + scale multiply + nibble/int8 reshape per K-block),
   (b) three dot products per K-block instead of one, and (c) a Triton
   tile schedule vs rocBLAS's tuned one. A policy switch to dense bf16
   prefill GEMMs abandons all three — which is precisely the numerics
   question gate 2 must answer, not a free lunch.
4. **Decode is untouched by design** (measured bandwidth-bound in #644/#645
   framing; the compensated kernel does not even accept M=1 — decode stays
   on the FP32 scalar matvec path).

## Next (gates 2–5, unchanged in shape)

Gate 2 must now be posed against the **shape-dependent ~2.3–7.3× ceiling**,
not the 20–80× impression: does a dense-bf16/fp16 prefill GEMM path hold
the #70/#535 replay contract (84-target exact greedy), knowing the #601
evidence
(mixed precision does not compose bit-exactly — head-logit max_abs 2.62,
argmax flip at 1/10 rounds) and that the compensated path is *already* an
approximation whose own deviation measured 5.7e-3 max_abs here? The
compensated kernel's deviation from FP32-dequant reference (measured
above) is the numerics baseline any candidate must beat or match
contractually. Gate 3 (package duty-cycle/thermal re-measure) and gate 4
(one bounded 4K cell) follow only if gate 2 produces a contract that
survives.

## Honest boundary

Synthetic weights, four GEMM shapes, one host, one thermal epoch. The
state/scan/attention portions of prefill are not GEMMs and are not priced
here. No throughput claim beyond these microbench arms; nothing in this
run qualifies an executor mode change.
