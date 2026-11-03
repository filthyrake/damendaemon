# Split prefill admission: the aggregate-prefill bound on dense concurrency (#690)

The 27B dense's concurrency ceiling was aggregate-prefill-bound, not
decode-bound: the chunked scheduler admitted the whole batch into
prefill at once, so every prefilling slot shared the one-chunk-per-
decode-step stream and **every slot's first output waited on all slots'
prefill** — C4 first outputs 502–528 s ≈ 4 × 125 s; the C5 projection
(~630 s > the 600 s gate) was arithmetic that turned out to be
**policy-refused anyway** (compensated Q4 admits at most four slots —
a measured memory bound, NOT overridden; 4×4 GiB KV + ~17 GiB packed
weights + MTP head ≈ 38.5 of 44.5 GiB).

## Fix (`split-prefill-690`)

`DAMENDAEMON_PREFILL_ADMISSION` bounds concurrent prefill lanes on the
chunked path (default `'1'` post-qualification; `'0'` = the legacy
batch-wide round-robin, the A/B control). At bound 1 a queued request
joins only when the prefill lane frees: the incumbent's first output
stops waiting on later arrivals' chunks, the later prefill overlaps
incumbent decode one chunk per decode step (the existing interleave,
unchanged), and requests beyond the bound wait in the FIFO queue (queue
timeout applies; the **post-admission SLO clock starts at admission**,
per the #59 contract). Scope: `_chunked_group` — the qualified
chunk-1024 path; the 35B whole-prefill serves and non-chunked paths are
untouched.

## Measured (A6000, guarded, one invocation per phase; C4 simultaneous — 4 × 32K, 27B Q4 spec serve, chunk 1024, MTP, batch/queue 4, queue-timeout 900)

| | legacy (control) | split | delta |
|---|---|---|---|
| first output, client-observed | 504.9 / 523.9 / 527.0 / 530.4 s | **129.7 / 268.8 / 419.3 / 576.2 s** | earliest 3.9× |
| **first output, post-admission** | 504.9–527.3 s (every user) | **129.7 / 138.9 / 150.0 / 156.2 s** | **worst user 3.4×; every user's SLO clock reads ~2 min** |
| queue wait | 0 / 3.1 / 3.1 / 3.1 s | 0 / 129.9 / 269.4 / 420.1 s (FIFO) | moved to capacity |
| request totals | 771.5–775.3 s | 730.5–779.2 s | unchanged — the overlap is free |
| outputs | — | **byte-identical digests, 4/4 slots** | scheduling changed, not one token |

Gates (declared on #690 before launch, amended once before any
measurement — see below): both phases 4/4 complete 512/512, zero errors,
no queue-timeout rejections ✓; split post-admission ≤ 600 s all slots ✓
(4× margin); split earliest client-observed ≤ 300 s ✓ (129.7); legacy
aggregate-bound signature ≥ 450 s tight spread ✓ (504.9–530.4,
spread 25.5 s — reproducing the #685 record's 502–528 s); slot-0 pinned
corpus digest reproduced ✓; per-slot digests byte-identical across
phases ✓.

Honest notes:
- The post-admission drift 129.7 → 156.2 s is the measured cost of
  overlapping each later prefill with more incumbent decode (one chunk
  per decode step) — small, and the totals show the aggregate is not
  paid back later.
- The queue wait is the capacity surface the bound moves: at C4 the last
  user waits 420 s in queue (< queue-timeout 900; the C-series 600 s
  timeout would bound simultaneous admission to ~4 users — at the
  policy's own slot cap, so no capacity is lost).
- The split phase's totals include slots finishing EARLIER than legacy
  (730.5 vs 771.5 s) — the staggered admission lets early slots stream
  while later prefills run.

## Retained findings (disclosed before measurement)

Three launcher iterations never reached measurement, all 400s at the
warmup request after ~175 s of weight load, root-caused with a
traceback-printing diagnostic patch on the staged /tmp copy (never the
repo):
1. Plain `damendaemon serve`: the Q4 production policy caps context at
   512 — the 32K Q4 cells must launch through the A6000 Q4 context
   probe (`hybrid_q4_a6000_probe serve`, the qualified C-series path).
2. The decisive one: **"experimental hybrid precision requires … at
   most four slots"** — the compensated Q4 precision policy admits
   batch ≤ 4 (measured memory bound). The C5 cell was withdrawn to the
   qualified C4 shape; the gate set was amended on #690 before any
   measured run. The policy bound is NOT overridden.

CPU contract (`tests/test_scheduler.py`, `SplitPrefillAdmissionTests` +
the legacy-pinning update): bound-1 chunk ordering (the incumbent's
whole prefill precedes the second request's first chunk; decode
interleave preserved), no starvation at 3 requests / batch 2 / bound 1,
env validation, default-is-split, and the legacy round-robin pinned
explicitly as the control.

## Artifacts

- `split-cell.json` / `g690a.json` (+ serve log) — phase split, all gates
- `legacy-cell.json` / `g690b.json` (+ serve log) — phase control, aggregate bound confirmed
- `runner.py` — the A/B verbatim
