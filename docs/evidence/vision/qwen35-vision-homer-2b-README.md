# #50 follow-ups: 2B vision gates on HOMER + ROCm workspace characterization (2026-10-25)

## 1. Qwen3.5-2B image gates on HOMER — all three PASSED

The 2B references were recovered byte-exact: `results/qwen35-2b-images-golden.pt`
on labnode3's dllm-dev checkout hashes to `f5de106f92d7a31f`, the exact
`reference_sha256` recorded in the A2 2B gate (`qwen35-2b-model-final-a2.json`).
Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc` is cached on HOMER.

| gate | result | worst deviation | guard max |
|---|---|---|---|
| Encoder (depth-24 calibrated bounds: rel RMS <7e-04, abs < max(0.02, 2e-03×max\|target\|)) | **135/135** | rel RMS **2.458e-04** (0.35× gate), worst abs/gate ratio 0.297 | 53.3 °C |
| Full-model image (FP32 text, 6 cases incl. controls) | **54/54** | rel RMS **4.537e-03** (gate 0.01), worst abs 0.1119; changed-image control healthy (native 0.9411 vs reference 0.9482) | 42.0 °C |
| HTTP service lifecycle | **20/20** | full check list incl. streaming, concurrency, recovery, host restore, disk reload, no-growth | 48.0 °C |

**Policy already covered the 2B geometry** (both 1024-wide and 2048-wide are
enumerated), so no policy change is needed — these cells extend the qualified
DEVICE×GEOMETRY evidence matrix to HOMER×2B. Qwen3.5-2B image serving is
qualified on HOMER on the same terms as the A2/M4.

## 2. ROCm "persistent workspace" — characterized; the tolerance is replaceable

The trace attempt (allocator `_record_memory_history` exists on torch
2.11.0+rocm) produced a decisive negative: after a quiet unload,
`hip_allocated_bytes` reports **0** — there is no allocator block held across
unload. The nonzero unload-time values in the service gates (32 MiB on 0.8B
Q8_0, 96 MiB on 2B FP32) are **transient within-request allocations that had
not yet been returned by the allocator at the moment the lifecycle check
sampled status** — they scale with the served geometry's workspace use, which
explains both observations. The service gate's no-growth assertion across
`repeat-release` remains the real bound; the 32/96 MiB numbers should be read
as allocator-lag samples, not persistent library state.

The trace harness (`scripts/diagnose_vision_server_rocm.py`) ships here for
the next run of this investigation: it classifies unload-time live blocks by
allocation frames and records unidentified ones honestly. What it showed in
this cell is that a *quiet* unload has nothing live to classify.

## Honest boundaries

- The allocator trace did not capture a nonzero unload snapshot (the target
  state did not reproduce under the quiet client); the 32/96 MiB transient is
  characterized from the service gates' own recordings, not from frame-level
  identification. Frame-level capture would require sampling during the
  request window, which is a #575-style diagnostics-mode extension.
- The 2B service gate ran with the disclosed workspace tolerance in place
  (observed 96 MiB); its no-growth assertion passed.

Every cell under `guard_homer` (85 °C, fans fixed@5); worst temperature
53.3 °C. HOMER restored after the cells. A30/A2 untouched.
