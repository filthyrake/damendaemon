# #536 planning precheck: two-rank A30+A2 Q4 split for Qwen3.8-27B

CPU planning only (no device allocation, no execution, no performance claim):
the exact production planner the pipeline build uses — `choose_stage_splits`
with actual Q4_0-packed weights and FP16 scales — against the pinned
Qwen3.8-27B `1d4bf0f2` checkpoint and realistic per-device budgets
(A30 22,528 MiB / A2 13,312 MiB, i.e. the #70 guard-era envelopes minus
reserve). Run on the desktop host (107 GiB available RAM) via
`scripts/experiments/plan_q4_split.py`; the Q4 loader policy's A30 identity and
context<=512 gates are substituted planning-only, the same way the #68/#70
context probe substitutes them for bounded execution.

## Results

| context | stage 0 (layers 0–46) | stage 1 (layers 46–64) | headroom |
| --- | --- | --- | --- |
| 2048 | 12,609 MiB (56% of budget) | 7,452 MiB (56%) | ~9.6 GiB / ~5.6 GiB |
| 8192 | 13,113 MiB (58%) | 7,812 MiB (59%) | ~9.1 GiB / ~5.3 GiB |

Packed weights 15.8 GiB (scales counted separately: ~1.05 GiB stage 0 +
~0.48 GiB stage 1). FP16 KV+state grows ~3.5 GiB per 10K context total
(16 full-attention layers x 4 kv-heads x 256 dim), so:

- Gate 2 (2K two-rank qualification): fits with >14 GiB combined headroom.
- Gate 3 (toward 8K, even 32K): FP16 KV at 32K (~4.3 GiB total) still fits;
  with #535's Q8_0 KV (~1.06 GiB total, measured) it fits with room to spare.
  The planner's context cap (2048) and the policy are the binding constraints,
  not device memory.

## What landed with this evidence

`issue-536-quant-pipeline`: the hybrid CUDA pipeline now admits packed Q8/Q4
weights behind `DAMENDAEMON_EXPERIMENTAL_HYBRID_PIPELINE_QUANT=1` (Q4 requires
compensated precision and builds Triton stages — the qualified A30 numeric
path); `shard_model` carries per-stage scale tensors; `stage_budget` counts
scale bytes; the build threads scales through planning. Unquantized default
contracts are unchanged (all existing pipeline/stage/placement suites pass).

## Not claimed

No GPU execution happened for this evidence. The gate-2 two-rank qualification
(exact logits vs single-rank control, HTTP lifecycle, per-stage memory on
device) is the next bounded step; the checkpoint still needs staging on
labnode3 (8 GiB free on /home — /tmp has room).
