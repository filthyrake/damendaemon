# #63 gate 3: serve-mode lifecycle + display headroom on the B580 (passed, 2026-10-18)

## Result

The queued serve runtime (`damendaemon serve`) ran on the B580 at the XPU
bring-up limits (Qwen3.5-0.8B FP32, context 2048, batch 4, no graphs) and
served a concurrent batch of four distinct prompts through the scheduler with
correct outputs (Paris / Yellow / `17 * 23 = 391.` / counting), 0.7-1.4 s
each, plus a warmup. Complete lifecycle: health, admission, generation,
telemetry.

- XPU allocator after the concurrent round: 3.29 GB allocated / 3.33 GB peak /
  3.35 GB reserved — comfortably inside the 12 GiB device with the 2 GiB
  display reserve while the desktop display was actively in use.
- Runner: `scripts/experiments/xpu_serve_gate.py` (health wait, warmup,
  concurrent threads, pre/post `/v1/models/status` snapshots). One runner bug
  fixed en route (the XPU env vars were built but not passed to the server
  Popen) and one gotcha recorded: every XPU process needs the local Level
  Zero loader via `LD_LIBRARY_PATH` — without it the serve child segfaults at
  device init.

## Scope

Gate 3 only: small fixture, FP32, context 2048, batch 4, single device.
Wall times are capability evidence on a shared desktop, never performance
claims. Still open under #63: quantized execution on XPU (the policy is
FP32-only by design during bring-up), context beyond 2048, matched baselines,
concurrency SLOs.

## Files

- `serve-gate.json` — gate report (rounds, statuses, headroom)
- `serve-gate.serve.log` — serve log
