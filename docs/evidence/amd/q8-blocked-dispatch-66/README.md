# #66 full-model gate: thermally paced blocked-dispatch ABBA (2026-10-24)

## Verdict

**PASSED — every measured pass cleared all 84 numerical targets, and the
blocked reduction removes the batch-scaling penalty in the full Qwen3.8-27B
Q8 model: 11x faster decode at batch 4 and 6.1x at batch 2, parity at
batch 1.** The one-candidate cycle is complete; integration is the only
remaining checklist item and is recorded as an explicit decision below.

## Run

HOMER Radeon 8060S (gfx1151), guard_homer hard cutoff 85 °C (max observed
83.0, sensors `amdgpu edge` + `k10temp Tctl`), fans fixed@5 verified
4.2k/4.3k/1.8k RPM, power capped 45/45/45 W via ryzenadj for the run and
restored to 120/140/120 afterwards, 28 °C idle at teardown. Tree
`issue-66-thermal-paced-gate` (`13067e35`, the frozen 84-target harness).
Thermal pacing (soft stop 60 °C pilot / 55 °C full run, same hwmon sensor
scan as the guard) inserted 61.5 s of idle across the full run; timed
windows still bracket exactly one synchronize/event/synchronize.

Storage q8_0, state FP32 (`DAMENDAEMON_HYBRID_PRECISION=float32`,
quantization env pinned and asserted), compensated BF16 prefill in both
arms, graphs off, 4 slots, trace
`qwen38-27b-q8-cpu-fp32-packed-t8-golden512.pt` (29 events / 84 targets,
FP32 reference). Arm A is October's arm A unchanged (FP32 matvec rows ≤4,
compensated rows ≥5); arm B additionally dispatches the merged blocked
matvec (`q8_blocked_matvec.py`) for one to four rows.

## Results (decode medians, seconds/step)

| pass | arm | batch 1 | batch 2 | batch 4 |
|---|---|---|---|---|
| A1 | FP32 | 0.1645 | 1.0721 | 1.9012 |
| B1 | blocked | 0.1649 | 0.1709 | 0.1733 |
| B2 | blocked | 0.1667 | 0.1755 | 0.1757 |
| A2 | FP32 | 0.1660 | 1.0716 | 1.9019 |

- batch 4: 1.901/1.902 → 0.173/0.176 (**11.0x**)
- batch 2: 1.072 → 0.171/0.176 (**6.1x**)
- batch 1: tied (0.165 vs 0.165-0.167)

Numerical gates: rel RMS < 0.01, max abs < 0.5, greedy match — all 84
targets in all six passes. This is the internal dispatch comparison the
issue asked for, not a matched vLLM/llama.cpp competitive result.

## Thermal workarounds that made this runnable

The unpaced 16 GiB fill and CPU Q8 conversion had been the sole blocker
(#74, post-pad verdict). Three stacked measures, all software:
1. paced CPU quantization and residency fill (soft-stop hysteresis on the
   guard's own sensors),
2. a cool-down gate before every replay event — the die-edge spikes ~20 °C
   inside each measured event, so events must start from a cool floor
   (80 forced an early kill at 86; 60/55 held 83 max),
3. 45 W power caps during the run (user-suggested; slows the ramp).

Run wall time: 375 s for the full 6-pass gate. The host is a thermally
sensitive APU; nothing here changes the #74 verdict that the die-edge
path needs physical service for sustained unpaced serving.

## Integration decision (checklist item 4)

Deferred for a separate change: the candidate is verified at full-model
scope and its evidence is complete, but integrating the blocked dispatch
into `damendaemon` serving (rocm_policy admission for 1-4-row Q8 decode)
touches the qualified serving path and deserves its own issue cycle with
service-level gates. Recorded as the follow-up rather than silently
dropped.
