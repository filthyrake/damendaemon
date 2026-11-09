# GLM-5.3 q4 fleet execution — rung 4a: selected-expert streaming (#719)

2026-11-08, rocky10-desktop A6000, guarded (thermal 90 / reserve 0 / 1500 s).
The fleet cell's device-memory mechanism: whole-stage device residency is
impossible at the rung-1 q4 projection (395.93 GiB vs ~109 GB fleet VRAM),
so the qualified pattern is **route on device, stream only the top-8
selected experts (plus shared) per batch from host-resident q4 banks**.

## Result — ALL GATES PASS (attempt 2, `evidence2.json` / `guard2.json`)

Real layer-6 weights; all 256 experts + shared packed once into host RAM
(5.08 GiB q4 bank); three sequential device phases (resident-q4 oracle →
shipped-fp8 oracle → streamed path) with explicit frees so at most one bank
is ever device-resident.

1. **Streamed == resident q4**: max_abs **0.0** — bit-identical output; the
   streamed path copies the identical packed bytes and runs the identical
   kernel path with the identical `grouped_routes` accumulation order ✅
2. **Streamed vs shipped-fp8 oracle**: rel RMS **0.11969**, max_abs 0.071 ×
   absmax (standing ≤0.15 envelope) — and EXACTLY rung 2's FFN-parity
   numbers to the last digit (same weights/seed/order; an independent
   determinism confirmation of the whole chain) ✅
3. **Transfer bounded**: **0.91 GiB streamed** (44 selected experts +
   shared for 8 tokens × top-8) vs the 5.08 GiB host bank — ~5.5× less
   device traffic than residency, and the bound holds by construction as
   top-8/256 ✅

Peak device reserved 9.73 GiB (one bank + streaming temporaries). 45
distinct experts selected across the 8-token batch (64 routes). 188.7 s.

## Attempt 1 retained (`evidence-attempt1.json` / `guard-attempt1.json`)

Honest sizing bug: the first version held the resident-q4 AND fp8 oracle
banks simultaneously under a 12 GiB allocator cap and OOM-refused at 154 s
(mid fp8 build, 11.3 GiB allocated). The fix (sequential phases + explicit
frees, 26 GiB cap) changes only when tensors live, not what they are —
gate semantics untouched.

## Significance for the lane

With this rung, every mechanism the full 4-host fleet cell needs is
individually qualified on real hardware: q4 packing + drift (rung 1), CUDA
executor parity (rung 2), ROCm executor parity (rung 3a), the cross-vendor
channel + chained execution (rung 3b), and now host-resident banks with
selected-expert streaming (rung 4a). #719's remaining work is the assembly:
the 78-layer 4-host chain (labnode3 / rocky10 / 3090box / homer) with
host-RAM stages and this streaming mechanism, gated on full-model generation
parity vs the #56 rung-4 reference oracle.
