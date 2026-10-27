# TokenGate 32K payoff cell: gate-adapted Qwen3.5-2B hybrid (#643)

Date: 2026-10-27. Host: rocky10-desktop idle RTX 3090 (24 GiB, driver
615.71.09, **280 W power cap held for the whole run**; peak draw ~272 W,
peak temp ~45 °C). Parent: #639 (decision record, GO at 2B scale); this is
child D — the owner-gated payoff cell, run exactly per the pre-launch
declaration recorded in the #639 decision comment.

The **first honest end-to-end evaluation of the released hybrid checkpoint
at 32K**: #640's hybrid arm was blocked by three upstream bugs (now patched
in the clone, cumulative #640+#642 set); nothing before this cell had
measured the paper's headline hybrid claim.

## Setup

- **Checkpoint**: `liana313/tokengate-niah-qwen3.5-2b` (HF snapshot
  `b1fe3e72b9896efc821fbfa02a19fcded4de6d77`; `gate_step18000.pt` sha256
  `a86e85aa…3ae6e25`, `lora_step18000.pt` sha256 `f93cc736…1a1c1a5b6`) —
  the paper's own benchmark artifact. Base: Qwen/Qwen3.5-2B revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- **Data**: the 32K length-generalization tier of NIAH-Kamradt
  (`niah_kamradt_32k.jsonl`, bodies ~32k tokens; checkpoints trained on the
  16k tier — the 2× length-generalization setting upstream's own demo uses).
  First 16 rows (deterministic; recorded convention).
- **Path**: the patched upstream harness (`generate_batch`,
  `prune_below_gate=True`, flex backend) — the qualified evaluator per
  #642's control ladder. The base arm runs plain HF generate on the same
  rows, with the tree's mandatory `patch_model_with_flex_attention` prep,
  and its TTFT is a bare-forward prefill wall with `logits_to_keep=1`
  (matched to the gate path's last-position-logits prefill).
- **Containment** (per the #643 gates): bounded 16 rows, absolute 120 s
  per-row deadlines (zero fired), atomic reports, one measured attempt per
  configuration, artifacts below. Runner: `tg643_cell.py`.

## Results (16 rows, greedy, 32 generated tokens)

| arm | acc | TTFT mean | speedup | peak alloc GiB | needle kept |
|---|---|---|---|---|---|
| base Qwen3.5-2B (control) | **1.000** | 3.038 s | 1.00× | 7.56 | — |
| gate-adapted, no prune | 0.938 | 3.211 s | 0.95× | 7.86 | 100% |
| gate, K=2000 | **0.938** | 0.417 s | **7.28×** | 7.27 | 76% |
| gate, K=500 | 0.875 | 0.305 s | 9.96× | 7.27 | 68% |
| gate, K=100 | 0.500 | 0.278 s | 10.93× | 7.27 | 51% |

(`tg643-cell.json` + `tg643-base-refit.json`; base TTFT/peak from the
refit — see the measurement-artifact note below.)

### What the numbers decompose into

- **Pruning at K=2000 costs zero accuracy** (0.938 = the adapted model's
  own no-prune score) for a **7.28× TTFT reduction** — the paper's
  headline mechanism, reproduced end-to-end on our hardware at the paper's
  own setting (its claim for this model: 3.8× at 32K NIAH; we measure
  above that, flex backend, single 3090).
- **The accuracy gap to the base model (1.000 → 0.938) is the adaptation
  delta, not pruning**: one row of 16, and the no-prune arm pays it too.
  This mirrors #642's co-adaptation finding at smaller scale, but far more
  mildly (−6% vs −50%): the upstream-released checkpoint is much stronger
  than our bounded self-trained one.
- **K=500 is nearly free** (0.875, one further row); **K=100 is not**
  (0.500, needle retained only 51% of the time — the gate provably drops
  the needle at the deepest budget; per-row `needle_kept` in the JSON
  correlates with per-row acc).
- **The paper's peak-memory claim (1.5–3.4×) does NOT manifest at this
  scale**: base 7.56 → gated 7.27 GiB = **1.04×**, essentially flat. The
  mechanism is visible in the geometry: the hybrid has only 6 full-attention
  layers (8 KV heads, head-dim 128) ≈ 0.75 GiB of prunable KV at 32K, and
  ~4 GiB of bf16 weights dominate both arms. #640's dense-1.7B probe
  measured 1.70× because the dense model carries ~4× the full-attention KV.
  Recorded honestly: at 2B-hybrid scale, TokenGate is a TTFT play, not a
  memory play.

### Measurement artifact found and corrected (retained)

The first base-arm numbers (TTFT 3.069 s, peak 18.97 GiB, in
`tg643-cell.json`'s original base entry) were polluted: the bare-forward
TTFT probe materialized full-vocab logits over all 32k positions
(~15.8 GiB, ~0.3 s) — work the gate arms' prefill never does. Fixed with
`logits_to_keep=1` and the base arm re-measured alone
(`tg643-base-refit.json`: 3.038 s, 7.56 GiB). Both records retained; the
table uses the refit. Launch-path failures of attempts 1–2 (tensor-shape
bug; state-sentinel ids fed to the embedding — `apply_chat_template`
emits negative/>vocab sentinel ids that `split_state_span` strips) are
retained as `tg643-cell-attempt{1,2}.log` with their probe scripts.

## Verdict against the #643 gates

- [x] Pre-launch declaration (model/revision/K/chunk/guard/host/stop rules)
      — recorded in the #639 decision comment before launch.
- [x] Guarded run, bounded prompts, deadlines, atomic reports; one measured
      attempt per configuration; 280 W cap held; zero row timeouts.
- [x] TTFT, peak prefill memory, completion, thermal recorded.
- [x] Accuracy-at-budget at 32K scale through the qualified evaluator.
- [x] Honest outcome: **PASS at the 2B cell scale** — 7.28× TTFT at
      K=2000 with zero pruning cost, against a 1.000-accuracy base control;
      the memory claim refuted at this scale with the mechanism identified;
      K=100 degradation recorded with the needle-kept census.

## Explicit non-claims

No 27B qualification, no "fits" claim for any 32K workload that failed on
capacity (this is a 2B cell on a 24 GiB card with 7.6 GiB peaks), no
serving/SLO claim (single-request, batch 1), no concurrency sizing. The
A30 Q4 32K capacity closure (#70) and HOMER's #68 envelope are untouched by
this cell; what it does establish for them: the gated-prefill mechanism
delivers real TTFT at 32K on our stack with measurable accuracy budgets —
if a 27B-class gate-adapted checkpoint ever appears, the payoff cell shape
is proven, but the memory lever (the blocker for #70) does NOT materialize
on hybrid geometry at this scale.

Reproduce: `tg643_cell.py` (in the tokengate-640 clone root on
rocky10-desktop, its `.venv`, `TG643_REPORT=…`, optional `TG643_BASE_ONLY=1`).
