# #537 gate 1-3 — burst-class HOMER lane defined and one cell delivered

Date: 2026-10-19. HOMER idle before the run (load 0.00, GPU edge 40 C,
uptime 7 days). Envelope: `envelope.md` (written before the run, per the gate).

## Run

- Workload: `scripts/experiments/bench_q8_blocked_matvec.py` — the #66
  blocked-matvec primitive contract (one bounded cycle): 9 shapes, three
  projection paths (current FP32 full-width / blocked FP32 tiled /
  compensated BF16+FP32), numerical gate + 5 ABBA timing cycles.
- Launcher: `scripts.experiments.guard_homer --limit-celsius 80 --timeout 600`
  with `DAMENDAEMON_ROCM_MEMORY_GIB=28` (the #74 convention; the first
  attempt without it hit the default 8 GiB policy cap at the largest
  reference expansion — a policy cap, not device exhaustion, retained in the
  guard's history on the host).
- Fan procedure per AGENTS.md: `strix-fan-daemon` stopped, `fan-inference.sh
  on`, all three EC fans verified `fixed` level 5 (4409/4464/1494 RPM) before
  load; guard readbacks during the run 4373/4445/1852 RPM.

## Result: PASSED, envelope held

- `dd537-burst.json`: `passed: true` — 9/9 numerical cases inside the
  1e-4 relative-RMS / 0.01 max-absolute gate on the Radeon 8060S
  (torch 2.11.0+rocm7.13.0a20260425).
- `dd537-burst-guard.json`: phase `completed`, exit 0, ~4 s of owned load;
  no thermal trip (80 C cutoff), host memory watchdog clean (110.6 GiB
  available before / after).
- Post-run telemetry: GPU edge 35 C, fans restored to `auto`/balanced,
  `strix-fan-daemon` active again — state restored and verified.

## Scope and honesty

- One burst-class cell delivered; this does NOT lift the sustained-prefill
  block (gate 4): integrated long prefill stays forbidden until #74's owner
  decision. No performance claim from shared-host timings — the ABBA samples
  are comparison evidence for the #66 primitive gate only.
- The failed first attempt (policy cap, 5.8 s) is retained as
  `/tmp/dd537-burst-guard.json` on HOMER.
