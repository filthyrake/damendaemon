# MTP draft-head attention bounding: the 128K transient wall (#689)

The #686 ctx128k cell died at the first spec round: the MTP head's
attention over 131K keys attempted a 2.99 GiB allocation with 2.76 GiB
free. This change eliminates that transient and re-runs the 128K C1 cell.

## Root cause (measured, two steps)

The MTP block's attention carried an explicit boolean mask ("causal minus
the position-0 hole" — the block never sees a predecessor for the first
token). The first fix expressed that structurally as `is_causal=True`
over keys sliced `[1:end]` — **and the 128K re-run still OOMed with the
identical 2.99 GiB signature** (first output 933.5 s, hard OOM at
`mtp_propose`, frame on the new sliced-causal call itself). The mask was
never the load-bearing trigger: the 27B Q4 serve computes in **fp32**
(`compensated` → `compute_dtype = float32`); the fused SDPA backends
reject fp32, the math backend runs, and with `enable_gqa` it materializes
**q-head-expanded fp32 K/V** (24 heads × 131K keys × 128 dim × 4 B ≈
1.6 GiB per tensor) regardless of mask vs is_causal. The same mechanism
produced the recoverable 45–72-warning churn march during both runs'
chunked prefill (the `device.attention` pre-expansion hatch at q_len ≥
256, ~1.6 GiB per layer, allocator-recovered each time).

## Fix (`mtp-causal-689` @ 442c9e8f)

`_mtp_layer_forward`'s attention, admission on structure:
- **contiguous rows or single rows** (every production call site:
  `_mtp_fill_chunk` fresh/continuation chunks, `_mtp_prefill`,
  `mtp_propose`/spec seeds and backfills) → keys sliced past the hole,
  then **fp32 computes take the #681 grouped GEMM path**
  (`grouped_attention.attention`): pure-torch per-kv-head GEMMs, fixed
  256 MB scratch budget (~3.3 MB actual at the q_len=1 spec call), no
  SDPA backend selection, no enable_gqa expansion, causal bound
  `prefix = context − length` — exactly hole+causal for contiguous rows.
  The same production-proven surface the main model's compensated chunk
  prefill rides on CUDA and ROCm. **fp16 computes keep the fused kernel**
  (flash handles is_causal + GQA natively).
- **non-contiguous divergent rows** keep the eager mask (structural
  fallback; no production caller, the guard remains).
- `DAMENDAEMON_MTP_EAGER_MASK=1` forces the legacy masked path as the
  parity control.

CPU contract (`tests/test_hybrid_mtp_serving.py`, 4 new tests): sliced/
grouped route ≡ eager-mask route greedy streams, chunked-fill KV parity
across routes (atol 1e-4), env validation, NaN-poisoned position-0 hole
never read.

## Measured (A6000, guarded, single-device, one invocation per cell)

Runner unchanged: `scripts/experiments/hybrid_q4_long_context_c1_a6000.py`
(27B Q4 compensated, chunk 1024, MTP served, 512 greedy output tokens).

**64K control (final build; gates: 512/512, first ≤ 600 s, request
≤ 1200 s, digest byte-identical to the #686 record, peak reported):**

| | #686 record | final build | intermediate build (retained) |
|---|---|---|---|
| first output | 332.0 s | 332.8 s | 325.7 s |
| total | 399.9 s | **390.2 s** | 437.7 s |
| digest | 17e24c15… | **17e24c15… (match)** | 17e24c15… (match) |
| peak | 32.35 GB | **27.99 GB** | — |
| allocator warnings | 0 (per log) | **0** | 3 |

Digest byte-identical at both builds; the grouped fill path also removed
the intermediate build's decode-phase overhead (total 390.2 vs 437.7 s)
and 4.4 GB of peak (the record's pre-expansion churn).

**128K target (the #689 wall):**

| | #686 baseline (died) | is_causal build (died; retained) | final build |
|---|---|---|---|
| first output | 959.7 s | 933.5 s | 956.0 s |
| then | OOM 2.99 GiB, worker died | OOM 2.99 GiB, worker died | **512/512 tokens streamed** |
| request total | — | — | **1029.1 s** |
| decode phase | — | — | 73.1 s ≈ **7.0 tok/s** (spec engaged) |
| allocator warnings | 72 + fatal | 45 + fatal | **0** |
| peak | 40.8 GiB + refused | 40.8 GiB + refused | **35.14 GiB** |

`completed=False` on the runner's own gates — **the pre-declared honest
outcome**: first output 956.0 s > the 600 s gate. The capacity boundary
moved (this fix's claim); the latency one did not (prefill is untouched —
~930–960 s super-linear in prefix, exactly as the issue framed it:
memory-pass + latency-fail). Decode at 7.0 tok/s single-slot at 128K
shows spec rounds fully engaged (2 tokens per accepted forward required).

Digest `3347c6bd…` reported — no 128K baseline exists (both prior
attempts died mid-decode); spec exactness stands on the serving tests
(every emission is the main model's greedy argmax at its position).

## Retained findings

1. First 128K re-run (is_causal-only build): identical 2.99 GiB OOM —
   fp32 + math + GQA expansion is mask-independent. Report + guard JSON
   retained (`ctx128k-intermediate.json` / `g689b.json`); the full
   traceback is quoted in the #689 amendment comment. Its serve log (and
   the intermediate 64K run's) were overwritten by the subsequent
   launches' logs — disclosed; the reports and issue comments carry the
   failure record.
2. The 45/72-warning churn march of the old builds is gone (0 warnings,
   both final cells) — the grouped fill path also bounds the prefill-side
   hatch churn.
3. The remaining 128K boundary is prefill latency (956 s > 600 s gate) —
   #690's split-prefill lane is the family that has to move it; no
   claim is made here.

## Artifacts

- `ctx64k-final.json` / `g689c.json` — final-build 64K control + guard
- `ctx128k-final.json` / `g689d.json` — final-build 128K target + guard
- `ctx128k-final-serve.log` — the clean 128K serve log (0 warnings)
- `ctx64k-intermediate.json` / `g689a.json` — intermediate build control
- `ctx128k-intermediate.json` / `g689b.json` — the failed is_causal run
