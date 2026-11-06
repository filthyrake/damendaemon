# GLM-5.3 first execution (#56, rung 4)

2026-11-06, labnode3 (`dsv4f-venv`, CPU torch, 8 threads) against the staged
official checkpoint. Cell C: teacher-forced forward through REAL weights via
the production `GlmStageSlots` path — layer [0,1) (dense MLP + full-indexer
MLA) and the full sparse-MoE indexer group [6,10) (256 routed experts top-8 +
1 shared, 4 layers). Boundaries follow the indexer-group discipline from
rung 3.

## Gates — ALL PASS

1. **EXECUTION**: 32-token prefill through both stages; finite; shapes
   (1,32,6144) exact. layer0 420.06 s; layer6-group 2372.08 s (4 MoE layers).
2. **STATE PARITY**: prefill(31)+decode(1) last-position hidden vs full
   prefill(32) last-position hidden — dense/full-indexer path
   max|Δ| 1.12e-8, rel **4.11e-6**; sparse top-8/256 MoE path max|Δ| 2.09e-7,
   rel **1.26e-5**. Same magnitude family as GLM-5.3-Flash's 6.4e-6/1.2e-5.
3. **RELOAD DETERMINISM**: full reload of both stages reproduces prefill
   outputs **byte-identically**.

No logits (mid-model), no generation, no quality claim — reference
discipline. Peak cell RSS 82.0 GiB.

## Identity

Whole-directory model_id of the staged official checkpoint
(`cpu_snapshot_identity.fingerprint`, 760 GiB bound):
`7d4d796bc284fd18b7fd4dd839695271eae11b9013d25fb9274f2aa7c15ba0dd`
(first attempt aborted 'snapshot changed before identity read' — it raced
the download's .cache metadata settling; relaunched clean after quiescence).

## Performance datum (measured, not extrapolated)

One MoE-group layer ≈ 2372/4 ≈ 593 s per 32-token prefill on 8 CPU threads;
the dense layer ≈ 420 s. Full-model CPU reference is far slower than
GLM-5.3-Flash's linear-attention path (~90 min for a 14+2-token turn there);
the #56 topology cell budgets accordingly (context 128, few tokens).

## Files

- `cell-c.log` — full gate log; `glm56_exec_cell_c.py` — the cell;
  `fingerprint.log` — model_id output.
