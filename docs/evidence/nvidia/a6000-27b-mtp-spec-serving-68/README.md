# A6000 27B MTP speculative SERVING — PASS, decode 4.4x (#68 decode lever, 2026-11-01)

**Result: ALL DECLARED GATES PASS.** The #683-qualified dense MTP draft
head now serves in production `damendaemon serve`: packed spec rounds emit
two verified tokens per slot per forward, byte-exact, at 0.50x total and
0.23x decode wall-clock on the pinned 32K Q4 workload.

## Measured (guarded A6000, plain production serve, Q4 compensated 32K,
chunk 1024, pinned corpus 32,256-token prompt, 512 output, temperature 0,
two separately guarded phases; guard 0 each)

| | plain (head dropped) | spec (head served) |
|---|---|---|
| First output | 125.7 s | 128.9 s (MTP prefill fill +3.2 s) |
| Total | 365.6 s | **183.7 s** |
| Decode (511 tokens) | 239.9 s (2.1 tok/s) | **54.8 s (9.3 tok/s)** |
| Output digest | `3a02b8ce24eb8618…` | **byte-identical** |

Gates: EXACTNESS (digest equality) — pass; THROUGHPUT (total ≤ 0.85x
plain) — **0.502x**, pass. The corpus accepted ~100% (#683: 511/511), so
every round emitted two verified tokens.

**Honest measurement note:** the decode ratio is 4.4x, ABOVE the
structural ~2x ceiling of two-tokens-per-forward. The stacked mechanism:
the [1, 2] continuation forward is also CHEAPER per forward than the
plain eager length-1 decode step (0.21 s vs 0.47 s per forward), so
token-acceptance and per-forward cost both contribute. Both effects are
measured jointly on the shared card; the decomposition is not separately
claimed.

**Load-detection control (declared as host-weight delta):** the
/v1/models/status field read 0 in both phases for this residency path;
the head-load evidence is instead the prefill delta (+3.2 s of MTP KV
fill compute) plus the token-rate itself (two-per-round is only possible
with the head; a head-less phase measures 365.6 s).

## The first run's finding (disclosed, fixed, pinned)

The A/B's first spec run measured EXACTNESS pass but decode IDENTICAL to
plain (370.4 s): the chunked-prefill tier — the loop this serve shape
actually runs — was never wired for spec; only the plain and n-gram loops
were. Root-caused by reading, fixed structurally (every decode loop now
engages spec rounds when the head is present and requests are eligible),
pinned by new loop-engagement tests (`MtpChunkedSchedulerTests` in
tests/test_scheduler.py — a fake executor whose spec path asserts the
chunked loop calls spec_round, never falls back to decode_batch while
eligible, packs co-ready slots, and finishes mid-block cleanly; 78
passed + 26 subtests in the file).

## Machinery (branch mtp-spec-serving)

- Scheduler `_speculate_mtp`: packed [B, 2] optimistic rounds —
  [verified, draft] per slot, divergent per-row positions, per-slot SSM +
  conv snapshot/rollback on rejection, corrective blocks resolving
  in-call; stateless seeding between calls (no per-request GPU state).
- Chunked-prefill coexistence: the draft head's KV fills incrementally
  during prefill chunks (carried h_{j-1}; whole-prompt hiddens never
  materialized) — structurally equal to the whole-prompt fill (CPU-pinned,
  1e-6 float noise).
- Divergent-rows attention: explicit per-row causal mask branch; every
  existing uniform-row caller keeps its fast paths unchanged.
- Fallback: any ineligible ready request (temperature, adjustments,
  schema/tools) → plain packed decode for the batch.

Tests: tests/test_hybrid_mtp_serving.py (5 cases — spec == plain greedy
across forced accept/reject/corrective paths, deterministic block
semantics, packed divergent slots, chunked coexistence, loud refusals) +
tests/test_scheduler.py engagement cases. 201+ passed across the
touched batteries (1 known timing flake isolated green).

## Artifacts

| File | sha256 |
|---|---|
| mtp-serve-plain.json | f6273a91be6dab02061b81f640e1b9f9670dc79fbf7cd0f47006d2885c2d9240 |
| mtp-serve-spec.json | 55b0fd0cb41d8e2cb5800e1c7cc7382c116be5d6a3360e989c1730f42236aa4d |
| guard-plain.json | 5d1067af8013a59f6c0bf6da01da80d3dd6d40dfd1e930ed17d09a2f1f765b2e |
| guard-spec2.json | 4e93f4cbab7de397a832c82503d2be3ba1a98452d9fc32c3181002ed94033ff4 |
| runner.py | 814a554ac2c32aae673398d59c51334aba148412cb51381e8a29fef14b86463b |
| plain.log | 71e53f4d48cff712ab82a5ce06f8a3aa3ed856554b7b1c0480b74477713ea989 |
| spec2.log | 76533023bfd55cad621d0af75f7de72d3f7960cff1de179935d5b875f516ab15 |

(The first run's spec report is retained in the #68 comment thread as
the disclosure of the unwired-loop finding.)
