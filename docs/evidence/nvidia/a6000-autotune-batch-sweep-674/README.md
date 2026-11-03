# #674 — Auto-tune cells 1–4 + the cell-3 batch-sweep correctness anchor (35B, A6000)

**Question.** Can an opt-in auto-tune mode enumerate the policy-admitted
configuration region, measure candidates honestly under the guard, and
emit a pinned serve-consumable recommendation — without re-implementing
policy or enabling experiment opt-ins?

**Answer.** Yes. Cells 1, 2, 4 (CPU) + the cell-3 harness are delivered,
and the guarded correctness anchor reproduces the known C-series batch
ordering through the full enumerate → measure → rank → emit pipeline.

## The machinery (commits `1ab7e1e5`, `a1b26c5f`, `c8a93507`)

- **`damendaemon/autotune.py`** — the tunable inventory and the
  admission-aware enumerator. Every refusal is produced by the policy's
  own code (`hybrid_q4_policy.validate`, `hybrid_precision.select`, the
  MTP head's constructor gates, the executor's kernel/prefill gates),
  called in serve order against a stand-in descriptor. Ambient
  `DAMENDAEMON_EXPERIMENTAL_*` refuses the whole search (production
  region only); inert knob combinations are refused as search-space
  hygiene; per-candidate env is set and restored even on refusal.
  Cell 2: the versioned workload manifest (digest-pinned corpus, exact
  counts, bounded bytes) and the objective as a measurable predicate.
  Cell 4: the winner's version-1 serve model-config + env + flags.
- **`auto_tune_measure_a6000.py`** — one candidate per guarded
  invocation (a candidate IS a serve boot): manifest-validated MTP
  intent (checkpoint probe), digest-verified pinned corpus, warmup,
  measured round, incremental retained-on-crash row.
- **`auto_tune_rank.py`** — ranking against the declared objective;
  refuses incomplete rows; emits the winner + model-config.
- **20 CPU tests**: real qwen3_5_text family on simulated cuda/rocm
  surfaces, policy-verbatim refusal texts, env hygiene, manifest and
  objective contracts, grid ownership.

## The anchor (guarded, pinned manifest, #671 C-series reference shape)

Manifest: 35B `Qwen_Qwen3.6-35B-A3B` q8_0 headless (the checkpoint
carries mtp.* tensors; the head loads only under the explicit opt-in —
`model.mtp` is the search intent, validated by the probe), context
32768, the pinned C-series corpus (32,256 tokens, digest
`31836e96…be727`), 2 requests × 512 tokens at concurrency 2, objective
= maximize aggregate output tokens/s, gates: first output ≤ 600 s,
request ≤ 1200 s, exact 512 tokens, zero errors. Search values:
`max_batch: [1, 6]`, `graphs: [false]` (the C-series reference shape —
see the amendment).

| candidate | aggregate tok/s | p95 first output | tokens | errors | first output (per request) | request (per request) |
|---|---|---|---|---|---|---|
| batch 1 | 24.81 | 28.3 s | 512+512 | 0 | 7.8 / 28.3 s | 20.6 / 41.3 s |
| **batch 6** | **34.36** | **15.5 s** | 512+512 | 0 | 7.8 / 15.5 s | 29.8 / 29.8 s |

**All three declared gates pass**: both candidates complete 2/2 × 512/512
zero errors; batch 6's aggregate strictly exceeds batch 1's (34.36 >
24.81 — the C-series direction: admitted concurrency amortizes prefill
and decode); the ranker puts batch 6 first and emits the winner record +
`ranking.model-config.json` (version-1, `--model-config`-consumable)
with the exact environment and flags (`--batch-size 6 --no-graph`,
`DAMENDAEMON_HYBRID_PRECISION=float16`, …).

## Findings retained (all three failed/stalled launches)

1. **`row-batch6.json` (run 1)**: serve refused at boot — `queue size
   must fit batch size`. The harness sized the queue to the workload
   concurrency (2) below the candidate batch (6). Fixed: queue =
   max(batch, concurrency).
2. **`row-batch6b.json` + serve log (run 2, graphs on)**: the serve
   booted and **OOMed in the measured round** — 528 MB allocation vs
   278 MB free on the 44.5 GiB card. Real tuner-relevant finding: the
   6-slot 32K q8 shape already peaks at ~43.7 GiB without graphs, and
   the graph pools for batch sizes 1..6 tip it over. The enumerator
   admits graphs per policy; the measurement eliminates the candidate by
   OOM; the row ranks out with the errors disclosed — the honest
   elimination path working as designed.
3. **Amendment (declared before the rerun)**: the anchor's reference
   shape is the #671 C-series, which ran graphs OFF; the manifest was
   amended to pin `graphs: [false]` and BOTH candidates re-measured
   (`row-amend-batch1.json` supersedes the graphs-on `row-batch1.json`,
   which is retained as a finding — batch 1 with graphs on measured
   32.55 tok/s).

## Honest scope

- One measured round per candidate (the anchor checks machinery and
  ordering, not tight confidence intervals); per-request distributions
  are in the rows.
- The tuner searches the production policy region only — the 27B Q4
  C-series shapes (probe-only) are outside by design.
- Cell 5 (`--auto-tune` pre-serve calibration) is explicitly later, per
  the issue's own staging.
- No cross-hardware transfer claim: the evidence records the card, the
  corpus digest and the guard conditions that bound transferability.
