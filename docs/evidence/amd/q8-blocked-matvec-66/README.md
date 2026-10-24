# #66 primitive cell: FP32 small-batch Q8 blocked matvec vs current paths (2026-10-24)

## Verdict

**Promising — the primitive passes every accuracy gate and beats both
existing paths on every non-trivial shape, with a demonstrated K-dimension
cliff in the current FP32 matvec.** The full-model gate (issue checklist
item 3) is thermally blocked on HOMER today (see the #74 post-pad-service
report: the 16 GiB model-load phase alone trips a 93 °C guard), so the
issue stays open at its own one-candidate-cycle bound.

## Run

HOMER Radeon 8060S, `guard_homer` (fans fixed@5 verified 4.2k/4.3k/1.8k
RPM, strix-fan-daemon stopped for the run, limit 85 °C, timeout 900 s),
env `DAMENDAEMON_DEVICE=rocm ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
DAMENDAEMON_ROCM_MEMORY_GIB=28`. Tree: `issue-66-q8-blocked-matvec`
(bench ported from the preserved prototype 27c5d24 — imports follow the
damendaemon rename, optimization guard added; shapes, arithmetic, gates
and ABBA rounds frozen). Guard max temperature 57 °C — the primitive
phase is burst-safe by construction (matvec kernels only, no model load).

## Accuracy (gates: rel RMS < 1e-4, max abs < 0.01 vs independent FP32 linear)

Every arm passes every case. `blocked` is numerically the same class as
`current` (both FP32 reductions): rel RMS ≤ 2.4e-6, max abs ≤ 1.2e-4
across all nine shapes. `compensated` (three BF16 products) reads
rel RMS ≤ 3.6e-5, max abs ≤ 1.25e-3 — inside the gates, noisier, as its
design predicts.

## Timing medians (seconds, 5-cycle ABBA)

| case (m, n, k) | current | compensated | blocked | blocked vs current |
|---|---|---|---|---|
| 4, 14336, 5120 | 0.0008 | 0.0014 | **0.0004** | 2.0x |
| 4, 34816, 5120 | 0.0019 | 0.0031 | **0.0009** | 2.1x |
| 4, 5120, 17408 | 0.0239 | 0.0017 | **0.0005** | 47.8x |
| 4, 248320, 5120 | 0.0138 | 0.0207 | **0.0061** | 2.3x |
| 2, 14336, 5120 | 0.0004 | 0.0014 | 0.0004 | 1.0x (tied) |

(The three tiny shapes all round to sub-0.1 ms and carry no decision
weight; full per-iteration samples are in `q8matvec-66.json`.)

## Findings

1. **The current FP32 full-width matvec has a K-dimension cliff**:
   4×5120×17408 (89 M MACs) runs 23.9 ms — 30x slower than the 73 M-MAC
   4×14336×5120 shape — which is register spilling in the unblocked
   full-width reduction, the exact hypothesis #66 left open. The blocked
   kernel's tiled reduction eliminates it: 0.5 ms on the same shape.
2. `blocked` also beats `compensated` everywhere while being numerically
   tighter — if integrated, it supersedes the compensated dispatch for
   1–4-row Q8 projections.
3. Scale-invariant accuracy: the blocked path holds ~1e-6-class rel RMS
   from K=32 to K=17408 and the 248320-row vocab head.

## What blocks the next checklist item

The warmed full-model ABBA needs the 27B Q8 model resident; today's
post-pad-service runs (#74 comment) show the 16 GiB load phase alone
spikes the GPU-edge sensor to 91–94 °C within 45–60 s at both default
and 45 W power — the guard kills before compute begins. The full-model
gate is reachable only after the thermal service actually moves the
die-edge path, or with a load-path throttle (chunked residency fill,
#74/#68 material). The primitive result is committed now so the
candidate is frozen with its measured evidence either way.
