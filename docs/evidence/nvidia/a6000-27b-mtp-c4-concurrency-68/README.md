# A6000 27B MTP-spec 32K concurrency — C4 PASSES at 4 slots (#68, 2026-11-01)

**Result: the C4 gate passes under MTP speculation.** The C-series
concurrency bound moves from 2 slots to **4 slots at full 512 output** —
the cell the #68 decode lane existed to answer.

## Measured (guarded A6000, production serve, Q4 compensated 32K, chunk
1024, MTP head served, pinned corpus rotations, 512 output per slot)

**C2-spec (per-row variant, guard 0):** 512/512 both slots; first output
253.4/261.6 s; totals 413.5/413.9 s (~3.2 tok/s/slot) — within 0.5 s of
the packed-attention variant's first pass (413.0/413.4), so the OOM fix
costs nothing at B=2.

**C4-spec (guard 0, zero OOM):**

| slot | first output | total | output |
|---|---|---|---|
| 0 | 502.0 s | 788.0 s | 512/512 (3020 chars) |
| 1 | 521.1 s | 791.3 s | 512/512 (3018 chars) |
| 2 | 524.3 s | 791.9 s | 512/512 (3024 chars) |
| 3 | 527.7 s | 792.1 s | 512/512 (3028 chars) |

Gates: first output <= 600 s — pass (head-less chunk-1024 passed this
too); **complete request <= 1200 s — pass at 788-792 s where the head-less
cell clipped all four slots at the 1200.3 s wall with ~385/512 tokens**.

**Packed-4 decode rate: ~264 s / 511 tokens ≈ 1.94 tok/s/slot (7.75
aggregate)** vs the head-less ~0.55 tok/s/slot measured in both October
and today's chunk-1024 cell — **3.5x under packing**, with corpus
acceptance holding at 4 slots.

## Run 1 — the disclosed failure (retained as evidence)

C4-spec run 1 died ~28 s into decode: `torch.OutOfMemoryError: Tried to
allocate 2.96 GiB` inside the spec round's divergent-rows attention —
all four streams ended simultaneously with tiny outputs. Root cause: the
packed [4, 24 q-heads, 2, 32K] masked-GQA SDPA call falls to the math
path, which expands K/V to the full query-head count in FP32 (~800
MB/slot transient at 32K) — fits at B=1/B=2, OOMs at B=4. **Fix:**
divergent rows run per-slot at the decode branch's proven
[1, heads, q, end] geometry (same mask semantics, same FLOPs, bounded
transient). Also fixed en route: the q8_0 path's divergent branch
re-indexed read_kv's already-row-indexed views by slot id (`indexed=False`
now) — latent, unreachable in this cell's Q4 path, caught by re-reading.
c4spec.json (the failed run) and c4spec.serve.log (the OOM traceback) are
retained in this directory.

## Context ledger (all cells, one tree lineage)

| cell | first output | total | 512/512? |
|---|---|---|---|
| C4 chunk 127 (Oct, head-less) | 616-625 s | 1200.7 s clipped ~328 tok | no |
| C4 chunk 1024 (head-less) | 491-518 s | 1200.3 s clipped ~385 tok | no |
| C4-spec chunk 1024 (MTP) | **502-528 s** | **788-792 s** | **yes** |

The October finding's two named levers — "a faster executor (bigger
chunk, graphs)" and decode-side speculation — are both now measured; the
remaining unmeasured lever from the original finding is split prefill
admission (#59 SLO work).

## Artifacts

| File | sha256 |
|---|---|
| c2spec2.json | 0dcba50b27d21d825a1cb7b4934d7507772e156e34775d83c3a506b0c2e1b9bd |
| c4spec2.json | 1e550b3bc0fb61c041494534f62388cda3454161bc61b3896c047655149a9012 |
| guard-c2spec2.json | 138f2ba299d4f736b1022473c27528f6b9d678657205b842419522686745f26c |
| guard-c4spec2.json | 40bdc604d061948f90893bbfcb5a17c3afb0e83009b77562e12f70b0fc7276bb |
| c2spec.json (run-1 C2, packed attn) | d6d515fb7598f53ce2b0d1017bca0c507781da22498bd8ba55b3bc69a3888f45 |
| c4spec.json (run-1 FAIL) | 08aff26095307755640a7ed4ce6a4b9726d583792281879f504320b8e3779a1b |
| c4spec.serve.log (OOM traceback) | f780b6e51047346ecedb8025f50e900787a6ffbcdef5c1ca1206cd9873c550c4 |
| guard-c2spec.json | b964eb6bed6550a4f6e86bdc31e5cedb90433960510be9487ea9f6087fc338c3 |
| runner.py | 03beb1196adad0ebb995789abf5af4f829f690ac2f1ed5215fa6dcc8d02aabee |
