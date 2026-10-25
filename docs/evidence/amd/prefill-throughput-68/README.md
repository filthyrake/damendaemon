# #68 prefill-throughput diagnosis: kernel roofline + C1 envelope reachability (2026-10-25)

## Question

The qualified 32K C1 cell (#624) completed functionally at chunk 127 but missed
the time bounds (first output 1071.8 s vs ≤600; completion 1238.2 s vs ≤1200;
~30 tok/s prefill). The issue recorded that only real prefill-throughput work —
not chunk knobs — could close the gap. This cycle measured where prefill time
actually goes and whether any tile/power configuration reaches the bound.

## Cell 1 — projection tile sweep (bounded, 2 GiB cap, guard 85)

Swept 11 launch configurations (row_tile 32..128 × reduction_tile 32..128 ×
warps/stages) of the compensated Q4 prefill kernel on the frozen 27B shapes
(gate_up 34816×5120, down 5120×17408, qkv 14336×5120), rows=127, ABBA timing,
FP64 reference from ordinary Torch Q4 decoding.

**Result: the production 32/32/4/2 configuration is at the kernel's roofline.**
The best alternative (128/32/8 warps) gains 3-5%; larger reduction tiles bust
LDS (`OutOfResources`) on gfx1151. The hypothesized M-tile weight-traffic win
did not materialize — the kernel is dequant-ALU-bound, not bandwidth-bound.

**Numerics finding:** max-abs error is tiling-independent (identical 2.22e-05
across all configs on gate_up) because it is dominated by the BF16-compensation
approximation. The down projection (K=17408) **exceeds the frozen October 1e-4
max-abs gate in current production** (1.87e-4, rel RMS 3.1e-5) — a pre-existing
production property of the K=17408 accumulation, not introduced by tiling.

## Cell 2 — instrumented prefill breakdown (4K + 8K, default power, guard 85)

`hybrid_prefill_trace.py` around the context probe, production chunk 127:

- Projections ≈ **72% of synchronized prefill spans**; linear-attention ≈ 29%;
  full-attention ≈ 17% and **flat across prefix buckets** (no quadratic blowup
  at 8K).
- The October 79 s-projections trace predates the compensated kernel; the
  current runtime is materially faster per token (steady ≈ 18.2 ms/token at
  default power, ~55 tok/s).

## Cell 3 — C1 rerun under the revised envelope (default power, guard 85)

Motivation: default power showed 55 tok/s steady prefill at 74.0 °C max on the
8K probe, suggesting the ≤600 s first-output bound (32,256 tokens) was within
reach without the 45 W cap.

**Result: guard killed the run at 196.8 s — edge reached the 85.0 °C limit and
the host-memory watchdog fired during the ramp.** The 8K probe's thermal
behavior does not transfer to the 32K cache-load ramp: the sustained-ramp wall
(#74) is independent of steady-state prefill speed. Watchdog trigger was
contained; cleanup completed; host restored (fans auto/balanced, daemon active).

## Conclusion — C1 time bounds are not reachable on HOMER

Arithmetic bound: even at FULL default power with zero throttling, steady
prefill is ~18.2 ms/token → 32,011 tokens ≈ 582 s, leaving no margin for the
ramp, decode, or scheduling under the ≤600 s bound — and the full-power ramp
itself trips the guard in ~200 s. Under the only qualified envelope (45 W caps
+ pacing), measured steady prefill is 33.5 ms/token → ~1072 s. **No power-cap
or tile configuration closes the gap; the bound is hardware-bound.**

The 45 W + pacing recipe remains the only qualified 32K envelope (completes at
81.0 °C max). The functional half of C1 stands: 32,011 input + 512 output
tokens completed with correct streaming output.

## Remaining path (new scope, not another envelope probe)

The Triton compensated kernel is at its roofline; a fundamentally different
implementation (native WMMA/gfx1151 instructions or a Composable Kernel Q4
GEMM with FP32-compensated epilogue) is the only remaining prefill-throughput
path. That is new kernel-engineering scope, separately estimated.

Files on the host at /tmp/sweep68/: `guard6.{json,stdout}` (tile sweep +
numerics), `trace-guard.json`, `trace-serve.jsonl` (4K/8K spans),
`c1r-guard.json` (the 85 °C kill), `sweep.jsonl`, `trace68.log`.
