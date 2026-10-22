# Qwen3.5-2B vision admitted (#575)

Date: 2026-10-21 (UTC). Checkpoint `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Supersedes the scoping in
`qwen35-2b-scoping-README.md` (#573): the depth-24 intermediate gates are now
calibrated from the oracle device-spread population and the 2B geometry is
admitted for experimental image serving on NVIDIA A2 and Apple M4 under the
same limits as the 0.8B (context <= 1024, at most four slots,
`DAMENDAEMON_EXPERIMENTAL_VISION` opt-in).

## Gate calibration (oracle device-spread population)

The #571 bounds were calibrated on the depth-12 population. For depth 24 the
population is the trusted oracle's OWN CPU-vs-CUDA spread
(`qwen35-2b-vision-oracle-device-control.json`, 135 stage-checks, TF32
disabled): worst relative RMS 2.205e-04 (square block.22), worst
element-relative absolute 5.350e-04 (square block.23, max|target| 5,866).
House rule (3x worst member, floored at the old constant):

- depth 24: `relative_rms < 7e-04`, `max_absolute < max(0.02, 2e-03 x max|target|)`
- depth 12: unchanged (#571: 1e-04, max(0.02, 5e-04 x max|target|))

Verdict recomputed against all 540 recorded depth-24 checks (oracle spread +
A2 fp16 + A2 fp32-storage + MPS): zero failures. The relative-RMS gate remains
the binding systemic-drift check; at depth 24 a runtime would now have to
deviate 3x beyond the oracle's own device spread to fail.

## Encoder gates with the calibrated bound

- A2, guarded (`qwen35-2b-enc-calibrated-a2.json`): all five cases pass; the
  previously failing square blocks 19-22 pass on merit (worst 1.70e-04 vs 7e-04).
- Apple M4 MPS (`qwen35-2b-enc-calibrated-mps.json`): all five cases pass.

## Full-model gates

`qwen35-2b-model-final-a2.json` (guarded A2, refactored loader): 54/54 — all
greedy choices match the independent reference across 6 prompt cases
(including inverted-pixels, reversed-order, slot-reuse, text-only controls);
worst relative RMS 0.00488 (gate 0.01), worst absolute 0.1126 (gate 0.5).
Changed-image control: native logit delta 0.9425 vs reference 0.9482.

## Real HTTP service gate

`qwen35-2b-service-a2.json` (guarded A2, torch 2.13.0+cu130): PASSED, 20
lifecycle checks — independent image-chat completions for all six cases,
warm repeats, 4-way concurrency, streaming content+usage, client-disconnect
recovery with a changed image, malformed-input 400 recovery, second model
alias, host-restore (tier 'host'), disk-reload after release, repeat release
with no allocation growth. Vision residency accounting matches the geometry:
668,209,152 bytes via the new `expected_vision_bytes` (the 0.8B constant
204,977,152 reproduced exactly by the same function).

Two harness defects found and fixed on the way (runtime unchanged):

1. The service gate pinned `--host-gib 8` and hardcoded the 0.8B vision byte
   count. Two 2B residents (5.1 GiB each with vision weights) exceed 8 GiB, so
   the first alias was evicted from the host cache and the host-restore check
   measured 'disk'. The budget now scales with the checkpoint estimate (and
   the estimate itself runs under the vision opt-in — a bare call returns a
   text-only number and under-budgets). The byte pin is replaced by
   `expected_vision_bytes`.
2. The allocation-reference validator matched only the older
   `getNewWorkspace` frame vocabulary; torch 2.13 emits the persistent cuBLAS
   workspaces via `setWorkspaceForHandle`. Both retained blocks (2 x 8.5 MiB,
   trace in `qwen35-2b-retained-allocations.json`, produced by the
   allocation-diagnostics mode) are identified as library workspace; the
   2B's 17 MiB residual is smaller than the 0.8B's qualified 24 MiB.

## Policy

`damendaemon/vision_policy.py` now enumerates both text geometries (1024-wide
0.8B, 2048-wide 2B; 24 layers; qwen3_5_text). Everything else — MoE
(`qwen3_5_moe`), other widths/depths, non-Qwen — is still rejected; context
and slot limits unchanged; opt-in unchanged.
