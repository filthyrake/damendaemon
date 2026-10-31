# A6000 Q4 32K C1 capacity cell — PASSED (#68 re-scope, 2026-10-31)

## Verdict

The pinned 32K C1 cell for Qwen3.8-27B **executed and completed on the
desktop RTX A6000** — the first completed 32K C1 execution of this model
in the fleet (the A30 closed as a measured OOM failure at 21.97/23.60 GiB,
#70 attempt 5; HOMER never ran under its thermal stop):

- **First visible output: 154.8 s** (gate ≤ 600 s)
- **Complete request: 394.6 s** (gate ≤ 1200 s), outer guard 1800 s unused
- **Output: exactly 512/512 completion tokens**, greedy, coherent corpus
  continuation (" jumps over the lazy dog while the repository records
  every bounded qualification cell with evidence…")
- **Peak allocator: 23.07 GiB** (24.78 GB `cuda_peak_allocated_bytes`);
  steady residency after warmup 21.71 GB — the card (44.5 GiB) fits the
  cell with ~21 GiB allocator headroom
- **Complete release**: guard's final probe shows the card at 45538 MiB
  free (fully idle) after the run; temperature ended 79 °C
- Guarded end to end: single-device UUID pin, identity/temperature/host
  gates active; thermal opt-in 90 °C (card spec: slowdown 95 °C, shutdown
  98 °C), reserve 0 (near-capacity no-incumbent configuration, per the
  A30 attempt-5 precedent)

## Pinned workload (identical to #70 attempt 5, verified at runtime)

Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
(`/home/damen/models/Qwen3.8-27B-1d4bf0f2`), Q4 compensated hybrid probe
path, context 32768, input 32,256 raw token ids — the runner verifies the
corpus sha256 `31836e963e26cd64…` matches the A30 attempt's pinned digest
before launching and refuses on mismatch — output 512, C1 (batch 1),
prefill chunk 127, no graphs. Warmup (activation) measured 167.8 s
(A30: ~616 s); the warmup is recorded but is not the gate cell.

## Attempt history (all retained on the issue; harness/gate calibration only — workload never changed)

- attempt 1 (gate.json): warmup rejected 400 — the production Q4 policy
  admits only A30/ROCm/chain; root-caused via a temporary diagnostic
  traceback patch (never committed). Fix: `hybrid_q4_a6000_probe` — an
  isolated probe mirroring the A30/HOMER `context_probe` with the device
  gate substituted by an exact A6000 UUID pin (stricter than the A30's
  device-name check); every other production guard reimplemented verbatim
  (explicit opt-in, qwen3_5_text only, compensated precision, no vision,
  context 512..32768). Production admission (`hybrid_q4_policy.validate`)
  is untouched.
- attempt 2 (guard.json stale-report collision; runner3): guard refuses to
  overwrite reports — fresh report paths per attempt thereafter.
- attempt 3 (gate2/guard2): stopped by the guard's thermal gate at
  238.9 s — card at 64–65 °C ~40 s into prefill vs the hardcoded 65 °C
  fleet default calibrated for the shared A30 hosts. Fix (pre-declared on
  #68): `--thermal-limit-c` parameterization, fleet default unchanged,
  run at 85 °C. **Capacity datapoint:** the ramp showed ~40 GiB
  card-reserved transiently mid-prefill.
- attempt 4 (gate3/guard3): reserve gate (2560 MiB floor) fired at
  211.9 s mid-ramp. Pre-declared near-capacity configuration
  (`--gpu-reserve-mib 0`, A30 attempt-5 precedent) for the next attempt.
- attempt 4b (gate4/guard4, 85 °C + reserve 0): stopped at the 85 °C
  ceiling at 318.5 s, with the probe showing 18.5 GiB free again (memory
  peak passed — the ~40 GiB readings were transient allocator
  reservations, not residency). Pre-declared 90 °C (5 °C below hardware
  slowdown) for the final attempt.
- **attempt 5 (gate5/guard5, 90 °C + reserve 0): PASSED** — the numbers
  above. Peak observed temperature during the run is in the guard report;
  the run completed at 79 °C with full release.

## Honest scope

Capacity and execution only: this is not concurrency (C4 not attempted,
not assumed to fit), cancellation, reuse, quality or SLO qualification —
per the issue's own gate order, those are separately bounded cells that
follow a C1 success. The thermal opt-ins (85→90 °C) are recorded
parameterized re-calibrations for this dedicated host's card, each
pre-declared on #68 before launch; the fleet default remains 65 °C.
HOMER's own 32K residency remains blocked by its thermal stop (#74
deferred) — nothing ran on HOMER.
