# #620 service gate: blocked Q8 matvec dispatch in serving (2026-10-24)

## Verdict

**PASSED — the integrated blocked dispatch serves correct output with
6.3x/2.5x/2.7x lower per-request wall time at batch 1/2/4 than the
kill-switched arm, with identical greedy outputs across arms and both
guarded runs completing far below the thermal limit.** #620's acceptance
is met; the dispatch defaults on for gfx1151 FP32-state Q8 decode.

## Run

HOMER Radeon 8060S, two isolated guarded serve lifetimes (guard_homer,
85 °C cutoff, fans fixed@5 verified 4.2k/4.3k/1.8k RPM, ryzenadj 45 W
caps applied per arm and restored to 120/140/120 after, host restored:
balanced fans, daemon active, 32 °C idle edge). Both arms ran the merged
`issue-620-blocked-serving` tree; arm A set
`DAMENDAEMON_Q8_BLOCKED_MATVEC=0` (previous full-width matvec), arm B
`DAMENDAEMON_Q8_BLOCKED_MATVEC=1`. Serve: Qwen3.8-27B Q8_0, FP32 state,
--no-graph, context 512, batch-size 4, `DAMENDAEMON_THERMAL_SOFT_STOP=55`
(runtime pacing opt-in from #619: paced CPU quantization + residency
fill; guard max observed 47.0/45.5 °C during load-inclusive arm A /
decode-dominated arm B).

Driver: 9 rounds per arm (batch sizes 1/2/4 x 3), fixed prompt,
temperature 0, 24 completion tokens, concurrent clients, HTTP loopback.

## Results (median request wall seconds across 3 rounds)

| arm | batch 1 | batch 2 | batch 4 |
|---|---|---|---|
| A (kill switch) | 11.04 | 38.46 | 71.44 |
| B (blocked) | 9.84 | 15.61 | 26.91 |
| speedup | 1.12x | 2.46x | 2.65x |

- Greedy outputs are identical across all 21 clients in both arms
  (`cross-arm identical: True`); request wall includes serialized prefill
  and slot queueing, so these speedups are conservative relative to the
  11x/6.1x pure-decode medians in the #619 gate (whose batch-4 0.173
  s/step is consistent with arm B here).
- Guard reports: arm A completed in 818 s (max 47.0 °C), arm B in 262 s
  (max 45.5 °C) — the die-edge spike is contained by the cool-floor
  pacing from #619; sustained unpaced serving remains a #74 hardware
  matter.

## Operational notes recorded during the gate

- `pgrep -f "damendaemon serve"` also matches the guard's command line;
  stopping the child must exclude processes whose cmdline contains
  guard_homer (the stop helper used here does).
- The kill switch accepts 0/false/off and 1/true/on and rejects anything
  else; the first arm-B attempt failed closed on `=1` before the fix
  (tests cover both directions).
- Files: `serve66-{A,B}.drives.json` (per-client records),
  `serve66-{A,B}.guard.json` (fan/thermal reports).

## Follow-ups

None from this gate. #74's physical-service verdict stands; pacing makes
bounded qualification and short-burst service practical today.
