# #674 cell 5 — `--auto-tune` pre-serve calibration (0.8B, A6000)

**Question.** Can a single opt-in flag run cell 3's bounded search INSIDE
serve startup — before binding admission — and apply the winner to the
serving runtime, with the standalone harness's evidence artifacts and an
honest refusal when nothing passes?

**Answer.** Yes, delivered and live-verified end to end.

## What runs (guarded, one invocation)

`damendaemon serve --auto-tune tune.json --auto-tune-output …` on the
A6000 with the 0.8B hybrid (q8_0, headless, context 2048; the pinned
C-series paragraph at 1024 tokens, digest-verified inside calibration;
2 requests × 64 tokens at concurrency 2; objective = aggregate output
tokens/s; declared sweep `max_batch: [2, 4]`, every other axis collapsed
to the serve's own defaults).

## Result — all three declared gates pass

1. **Calibration inside startup** ✓ — the serve log carries
   `auto-tune applied in 12s — batch 4, graphs on, metric 393.7
   (aggregate_output_tokens_per_second)` BEFORE `damendaemon listening`.
   The artifact set is complete: ranking + winner, the
   serve-consumable model-config, per-candidate rows, refusals.
2. **The calibrated serve serves** ✓ — a completion request from the same
   pinned corpus returned exactly 64/64 tokens, zero errors, first
   output 3.0 s.
3. **Failure honesty** ✓ (CPU-pinned, restated): a failing candidate is a
   disclosed unmeasurable row; all-failed refuses to start (never a
   silent fallback).

| candidate | aggregate tok/s (direct-executor measurement) | outcome |
|---|---|---|
| batch 2 | 275.9 | ranked second |
| **batch 4** | **393.7** | **winner, applied** |

Zero refusals (both collapsed-default candidates admit on this surface).
Bonus consistency, reported not gated: the winning direction (batch 4 >
batch 2 at admitted concurrency) matches the C-series known ordering —
the same direction cell 3's anchor gated on the 35B.

## The machinery

- **`damendaemon/autotune_serve.py`** — `calibrate()`: manifest/serve
  consistency (alias, path, context, device), fresh-output enforcement,
  the cell-1 enumerator (production policy only), the cell-3 ranker,
  incremental per-candidate rows (retained on crash), cell-4 winner
  emission; `direct_measure()`: the qualified #671 C-series
  direct-executor pattern (sequential per-slot prefills + lockstep
  batched greedy decode — relative ordering is the decision surface; no
  sockets bind before calibration completes).
- **`damendaemon/cli.py`** — the `--auto-tune` / `--auto-tune-output`
  flags; the winner is applied before the runtime constructs (env vars
  are read at executor construction — load-bearing, identical to
  launching serve with the recommendation by hand).
- **11 CPU tests** (`tests/test_autotune_serve_674.py`): winner
  selection + artifacts, failure retention, all-failed refusal,
  gate-elimination vs refusal, manifest/serve match, fresh output,
  concurrency fit, empty search, and real CLI subprocess flag checks.

## Honest scope

- Machinery cell: the 0.8B qualifies the pipeline, not a performance
  claim; per-candidate executors reload weights (disclosed cost).
- Calibration measurement is direct-executor, not HTTP — ordering across
  candidates is the decision surface.
- Default remains fully manual: without the flag, nothing runs.

## Artifacts

`cell5.json` (runner report), `cell5.serve.log` (the applied line before
the listener), `tune-674-serve.json` (the pinned manifest),
`g674s.json` (guard), `artifacts/` (ranking, model-config, refusals,
both candidate rows).
