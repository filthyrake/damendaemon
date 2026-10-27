# Bounded gate adaptation on Qwen3.5-0.8B (#642) — evidence

Child C of the TokenGate epic (#639), downstream of #640 (evaluation) and
#641 (gate module port). Parent issue: #639.

## What was run

One bounded TokenGate adaptation cell on a single power-capped RTX 3090
(280 W limit, held for the entire 10 h 45 m run; peak 66 °C):

- **Base model**: Qwen/Qwen3.5-0.8B pinned at revision
  `2fc06364715b967f1860aea9cf38778875588b17` (local snapshot,
  sha256-verified download), identical 24-layer 3:1 hybrid map to the 2B
  (full-attention layers {3, 7, 11, 15, 19, 23}).
- **Recipe**: the upstream l1d lane verbatim
  (`train-scripts/niah-qwen3p5/run_e2e_mlpgate_pretrain_niah_iso_coadapt17_l1d_qwen3pt5.sh`,
  clone @`6e51aac`): gate at DeltaNet layer 1, bias on the 6 full-attention
  layers above, LoRA r=16 α=32 on q/v_proj of those layers, Muon + aux-AdamW,
  LogUniform(128, seq/2) phase-2 budget sampling + full-context sentinel,
  SWA on, K-eval 2000, seed 0.
- **Bounded edits vs the lane** (all recorded in the driver header,
  `scripts/experiments/token_gate_train_642_lane.sh`): 1 GPU instead of 8,
  grad-accum 4 (effective batch 4), 2 epochs instead of 20 (512 optimizer
  steps / 2048 micro-steps), warmup 50, eval freq 300 with 25 samples/group,
  grad checkpointing on, wandb offline. Everything else lane-verbatim.
- **Data**: NIAH-Kamradt regenerated with a train split
  (`--n-train 1024 --seed 0`); the 200 test rows are byte-identical to the
  #640 pool (verified). Tokenized + packed per the repo's own tooling for
  the 0.8B tokenizer.

Artifacts (host `rocky10-desktop`, run dir under
`/home/damen/642-runs/.../baseline/L1/`): all step checkpoints retained
(300..1800 + final + SWA). Best checkpoint by the run's own keep-best
criterion (`niah_heldout` at K=2000): **step 1200**
(`gate_step1200.pt` sha256 `c0e1c6edc9113eeb3130e2c01cf46269ea3c0f82369ad93099fe902094901f0a`,
`lora_step1200.pt` sha256 `01d982dc35d0993ea235ee5b4a403d7702d45004211596a29dfc67d127b81db0`).
Merged serving snapshot written to a distinct directory; base blobs never
modified (#564 lesson).

## Training result (upstream harness, in-training evals)

Wall: 10 h 45 m, 17.5 s/it steady, zero OOM, loss 2.86 → 0.0008.

Gated K=2000 accuracy (`niah` = 50 val rows / `niah_heldout` = 150 test rows):

| step | niah | heldout |
|---|---|---|
| 0 (init) | 0.220 | 0.113 |
| 600 | 0.040 | 0.040 |
| 900 | 0.160 | 0.153 |
| **1200** | **0.480** | **0.587** ← best (keep-best) |
| 1500 | 0.340 | 0.333 |
| 1800 | 0.580 | 0.467 |
| final (live) | 0.280 | 0.260 |
| final (SWA) | 0.340 | 0.420 |

Full-context baseline (no gate): 0.92 / 0.887. The gate genuinely learns
needle retrieval through a 2,000-token budget (0.11 → 0.587 heldout at peak),
but the bounded window (2 epochs, 1 GPU vs the paper's 20 epochs, 8 GPUs)
leaves it far from the full-context baseline, and the curve oscillates then
overfits late (train loss → 0.0004 while final eval drops to 0.26).

## First damendaemon-stack numbers (24 test rows, our executor, same 24 rows in every arm)

Convention: accuracy/drift per the #23 pattern — pruned forwards compared by
accuracy, never byte-exactness. Greedy, 48 generated tokens, identical
prompts across arms. TTFT = synchronized prefill wall time.

| arm | acc | TTFT mean | peak alloc |
|---|---|---|---|
| base, full context (control) | 22/24 (0.917) | — | — |
| adapted (merged LoRA), full context | 10/24 (0.417) | 13.60 s | 3.78 GiB |
| adapted, gated K=2000 | 0/24 | 3.02 s | 3.72 GiB |
| adapted, gated K=1000 | 0/24 | 2.28 s | 3.72 GiB |
| adapted, gated K=500 | 0/24 | 1.90 s | 3.72 GiB |
| adapted, gated K=100 | 0/24 | 1.61 s | 3.72 GiB |

TTFT through our executor's gated prefill: **4.5× at K=2000, 8.4× at K=100**
vs the same checkpoint's full-context prefill — the first damendaemon-stack
TokenGate speed number. Peak memory is flat across arms at this scale
(0.8B weights dominate; the paper's memory claim needs larger models to
manifest).

## Controls and the honest negatives

Three controls isolate every number (probe scripts archived alongside):

1. **Base through our executor = 0.917** — our scoring/greedy/decode harness
   is sound; the adapted-backbone drop is real, not a harness artifact.
2. **Merge fidelity: merged snapshot = live-LoRA model exactly** — full
   context 10/24 via our executor on the merged snapshot vs 10/24 via
   plain HF generate on the live-LoRA model. The LoRA merge is faithful;
   the co-adapted backbone genuinely lost full-context NIAH (0.92 → 0.42)
   at this checkpoint — the full-context sentinel did not preserve
   full-context behavior at this bounded budget. Recorded as a finding.
3. **Same checkpoint + same 24 rows through upstream's own (patched)
   harness: 5/24 gated at K=2000 with 12 empty generations** — the
   step-1200 checkpoint is genuinely weak on this 24-row subset (the
   in-training 0.587 was a 150-row average), *and* our #641 gated executor
   path still diverges end-to-end from the upstream harness (0/24 with
   degenerate counting output vs 5/24). Component parity (#641,
   drift-0.0 on all gate components) did not transfer to end-to-end
   inference parity; the divergence lives somewhere in the phase-2/decode
   integration (temperature/fold/KV-position semantics), not in the gate
   math. This is the open input to #639's go/no-go and blocks any serving
   claim (#642 was scoped to training + records, not serving).

## Upstream bugs found and patched this cell (ledger, cumulative with #640)

Five new blockers on the hybrid l1d lane, each with a minimal recorded patch
(`patch_642_*.py` in this directory, applied to the @6e51aac clone):

1. `train_e2e_gate.py` gated eval never passes `prune_below_gate` — the l1d
   hybrid gate has no KV layer under the gate → `StopIteration` at
   `gate_decode.py:147`. Patch: route lanes with no full-attention layer at
   or below the gate through `prune_below_gate=True`.
2. `gate_decode.py` decode loop passes the 4D kv mask to linear-attention
   layers → `apply_mask_to_padding_states` shape error
   (modeling_qwen3_5.py:234). Patch: `attention_mask=None` for
   linear-attention layers (no kv cache to mask).
3. The driver's kernel-default `case` globs only `Qwen/Qwen3.5-*` repo IDs —
   pinned local snapshot paths miss it. Patch: add the local-path pattern.
4. `train_e2e_gate.py` hard-imports wandb at train start (their own `train`
   extra, absent from the default env). Installed the locked 0.30.0,
   offline mode.
5. FlexAttention head-dim-256 default tiles compile a backward needing
   114,688 B shared memory vs the 3090's 101,376 B (A100-sized). Patch:
   `kernel_options` with the inductor `bwd_*` tile keys (documented in
   torch's own error path) — plain `BLOCK_M/N` only configure the eager
   kernel.

Also: `flash-linear-attention` is not in upstream's lock — the hybrid
DeltaNet path trains on the reference PyTorch fallback (upstream's own
configuration; recorded, not patched).

## Verdict against the acceptance gates

- [x] Bounded training driver, pinned seed/data/model, guarded, recorded.
- [x] Accuracy-at-budget record (upstream-harness curve + our-stack sweep,
      convention stated; no byte-exactness claim).
- [x] Measured TTFT/memory on our stack vs full context, same hardware.
- [x] Checkpoint artifact handling (digests pinned, distinct dirs).
- [x] Honest record of what did NOT hold: under-convergence in the bounded
      window, co-adaptation full-context loss, and the end-to-end executor
      divergence — recorded, not swept.

Closes #642 as a complete bounded cell with partial-convergence findings.
The #639 go/no-go inherits: mechanism trains and speeds up prefill on our
stack, but hybrid gated serving accuracy is NOT yet qualified (upstream
reference weak on this checkpoint subset; our executor diverges).
