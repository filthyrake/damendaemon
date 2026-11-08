# GLM-5.3 q4 fleet execution — rung 2: CUDA executor parity on the A6000 (#719)

2026-11-08, rocky10-desktop A6000 (44.5 GiB, pinned UUID `GPU-4adb9ddb…`),
guarded (`guard_shared_cuda`, thermal 90 °C, reserve 0, timeout 1500 s — the
#668 A6000 envelope; the cell is ~20 GiB peak, NOT near-capacity, disclosed
here). Official GLM-5.3 NAS original, layer 6: 256 routed experts top-8 +
shared expert, real router (bf16→FP32, recorded) and correction bias —
never packed (#668 selection-exactness).

## Result — ALL GATES PASS (attempt 3, `evidence3.json` / `guard3.json`)

Oracle bank: the shipped blockwise-E4M3-FP8 storage placed directly
(`e4m3fn_f32`). Candidate: the same weights packed to canonical Q4_0
(`q4_0_f16`) ON DEVICE, admitted by the widened uniform-bank admission. Both
through the production `CudaPackedGlmFeedForward`, batched composition, 8
seeded FP32 tokens at width 6144.

1. **Per-expert parity**: every expert the real router selected (44 routed
   IDs on the pinned inputs) + shared — all rel RMS ≤ **0.1464** (gate 0.15,
   pre-declared from rung 1's measured 10.24% worst weight-level drift at
   1.5×).
2. **Full FFN parity**: rel RMS **0.1197**, max_abs **0.071 × absmax** (gate 0.15).
3. **q4 composition equivalence**: batched vs per-token, max_abs **1.9e-6**
   (identical per-row arithmetic by construction).
4. **Memory**: q4 bank **5.457 GB** vs fp8 9.704 GB (**0.562×**); peak
   reserved 15.57 GB; guard completed, peak 60 °C, exit 0, 132.5 s.

No token, quality, throughput or SLO claim. Capability + parity record only.

## Retained findings (both attempts, per honest-measurement discipline)

- **Attempt 1** (`evidence1.json`, `cell1.log`, `guard1.json`): refused at
  construction — safetensors hands back native `float8_e4m3fn` codes, the
  cell asserted uint8. Fast honest refusal, no GPU time wasted.
- **Attempt 2** (`evidence2.json`): **the paired-oracle design caught a real
  decode bug** — the q4 packer's truth decode used `codes.to(float8_e4m3fn)`
  on uint8 bytes (a NUMERIC cast, uint8 153 → fp8 ~152) instead of the byte
  reinterpretation `.view(float8_e4m3fn)`. The garbage-packed q4 bank failed
  the parity gates at 317–320× rel RMS (worst per-expert 75.1, shared 319.6,
  FFN 317.1) while the oracle path was correct — exactly what the paired
  design exists to catch. Gates FAILED the cell; fix landed as `fba26a3c`
  (byte-view decode, pinned bit-exact against the production tile decoder
  `packed_tiles._e4m3` by a new CPU test).
- **Rung-1 drift record stands** (re-measured `rung1/rung1-q4-v2.log`,
  byte-identical numbers): the `.to()` bug was CELL-PATH-ONLY — rung 1 loaded
  native fp8 dtype tensors where `.to()` is a same-dtype no-op, so its
  published drift (8.1–10.2% rel RMS) was measured against the correct
  reference all along. No number changes; the incident and its scope are
  recorded here.

## Files

- `evidence3.json` / `guard3.json` — the passing cell (attempt 3)
- `evidence2.json` — the retained 300×-drift finding (attempt 2)
- `evidence1.json` / `guard1.json` / `cell1.log` — the dtype refusal (attempt 1)
- `cell2.log` — empty (attempt 2 exited cleanly through the gate failures)

Runner: `scripts/experiments/verify_glm_q4_a6000.py` (PR #721).
Pre-declaration: #719 comment 6727935540 (gates fixed before launch; no
threshold was touched at any point — the passing margins are the
pre-declared gates).
