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

## Chunk-1024 re-measurement (2026-11-01, after #681's grouped prefill)

Branch `c4-chunk-bounded` — the finding's own named lever ("bigger
chunk"), pre-declared on #68 before launch: exactly two cells, C2 then
conditional C4, chunk bounded 127..4096 in the runner (`--chunk`,
recorded in the reports). Everything else pinned as October.

**C2 @ 1024 — PASSED** (c2-1024.json / guard-c2-1024.json, guard 0 in 936 s):

| slot | first output | total | tokens |
|---|---|---|---|
| 0 | 257.2 s | 751.6 s | 512/512 |
| 1 | 248.7 s | 751.6 s | 512/512 (3018 chars) |

Peak allocator 25.12 GiB (vs 27.23 GiB at 127: the #681 grouped scratch
replaces the per-chunk SDPA temporaries). Warmup single request 171 s.

**C4 @ 1024 — first-output gate now passes, total gate still bounded out**
(c4-1024.json / guard-c4-1024.json, guard 0 in 1403 s):

| slot | first output | total (clipped) | chars |
|---|---|---|---|
| 0 | 491.5 s | 1200.3 s | 2296 |
| 1 | 510.9 s | 1200.3 s | 2263 |
| 2 | 514.4 s | 1200.3 s | 2258 |
| 3 | 518.3 s | 1200.3 s | 2254 |

Peak allocator 33.41 GiB, zero OOM, clean release. Aggregate prefill
improved ~1.24x (first output 616-625 → 491-518 s, all slots inside the
600 s gate the October cell failed). But all four slots were clipped
mid-decode by the 1200 s per-request wall at ~385/512 tokens.

**Honest correction to the October finding:** the "~0.8 tok/s/slot"
packed-4 decode estimate assumed 4.05 chars/token, derived by assuming
near-completion. The C2 cells measure this corpus at **5.90 chars/token**
(gate-c2.json slot 0: 512 tokens / 3020 chars; c2-1024.json identical).
Re-derived, both C4 cells decode at **~0.55-0.56 tok/s/slot (~2.2
aggregate)** — October ~328 tokens in 585 s, chunk-1024 ~389 in ~709 s.
The October conclusion (total gate fails at C4×512) stands, more
decisively: a full 512-token request needs ~500 s prefill + ~930 s
decode ≈ 1430 s. Chunk size moves prefill only; packed-4 decode is
unchanged, as expected.

**Updated finding:** on this executor, 32 K Q4 is 2-slot at full 512
output; 4-slot fits only shorter requests (≤ ~350 output tokens at the
measured rate) — or needs a faster decode path (graphs, speculative
decode, or split prefill admission under #59), not a bigger chunk and
not more memory.

## Honest scope

Capacity/concurrency only. Cancellation, reuse, quality and SLO remain
separately bounded cells per #68's gate order. HOMER untouched.
