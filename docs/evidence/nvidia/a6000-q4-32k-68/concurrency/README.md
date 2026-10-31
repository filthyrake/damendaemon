# A6000 Q4 32K concurrency cells — C2 PASS, C4 bounded out by time gates (#68, 2026-10-31)

Parent evidence: `../README.md` (the passing C1 cell, PR #659). Runner:
`scripts/experiments/hybrid_q4_32k_concurrency_gate_a6000.py` — same pinned
model/format/context/output/chunk as C1; N deterministic per-slot corpus
rotations (slot 0 verified to reproduce the pinned C1 digest at runtime;
distinct digests per slot, all recorded in the reports). Per-request gates
unchanged from C1: first streamed output ≤ 600 s, complete request
≤ 1200 s, per request. Guard: thermal 90 °C, reserve 0, outer 1800 s — the
pre-declared near-capacity configuration for this card, disclosures on
#68 before each launch.

## C2 — PASSED (gate-c2.json / guard-c2.json)

| slot | first output | total | tokens |
|---|---|---|---|
| 0 | 308.1 s | 805.0 s | 512/512 |
| 1 | 310.6 s | 805.9 s | 512/512 |

- Coherent greedy output on both slots. **Peak allocator 27.23 GiB** —
  the C1→C2 delta (+4.16 GiB) is exactly one 4.0 GiB KV slot, matching
  the config math (16 full-attn layers × 4 GQA heads × 256 dim × 2 ×
  fp32 at 32 K tokens).
- Observed 86 °C under two-slot load. Complete release (guard final:
  45538 MiB free).

## C4 — measured FAILURE on the per-request time gates (gate-c4.json / guard-c4.json)

| slot | first output | total (clipped) | tokens before clip |
|---|---|---|---|
| 0 | 616.2 s | 1200.7 s | ~478 (1934 chars) |
| 1 | 624.8 s | 1200.7 s | ~475 (1914 chars) |
| 2 | 619.9 s | 1200.7 s | ~476 (1919 chars) |
| 3 | 621.7 s | 1200.7 s | ~476 (1915 chars) |

- **Memory fits**: allocator 32.66 GiB steady at batch 4 (warmup), card
  ran at 42.4 GiB total with zero OOM; the C4 projection (~35.6 GiB
  peak) was not the constraint. Clean release after (45538 MiB, 80 °C;
  guard phase `completed` — the runner tore down honestly after the
  gates tripped).
- **Both time gates fail**: first output ~620 s across all slots
  (the pre-declared projection of ~619 s was exact — over the 600 s
  gate), and packed-4 decode at ~0.8 tok/s/slot puts a full 512-token
  request at ~1260 s (over the 1200 s gate); all four slots were
  clipped by the request limit mid-decode with coherent partial output.

## The finding

32 K concurrency on this executor (single A6000, Q4 compensated hybrid,
chunk 127, no graphs) is **throughput-bound at ~2 slots, not
memory-bound**: packed prefill (active for Q4 via the default
q4-stage-adjacent) did not improve total prefill throughput at C2
(first output doubled exactly: 308 s ≈ 2 × 154.8 s — first output waits
for all slots' prefill at ~fixed aggregate throughput), so C4's first
output scales linearly to ~620 s. Decode at 4-slot packing drops to
~0.8 tok/s/slot (~3.2 aggregate). C2 is the qualified concurrency level
at 32 K under these gates; higher concurrency needs either a faster
executor (bigger chunk, graphs — separate qualified changes) or split
prefill admission (SLO work under #59/#60), not more memory.

Per the pre-declaration on #68, this C4 failure is the measured finding
and is not a retry license; no setting sweep follows.

## Honest scope

Capacity/concurrency only. Cancellation, reuse, quality and SLO remain
separately bounded cells per #68's gate order. HOMER untouched.
