#!/bin/bash
# #642 bounded gate-adaptation cell — thin driver over upstream train-scripts.
# Base lane: train-scripts/niah-qwen3p5/run_e2e_mlpgate_pretrain_niah_iso_
# coadapt17_l1d_qwen3pt5.sh (the lane that produced the released
# liana313/tokengate-niah-qwen3.5-2b gate), clone pinned @6e51aac with the
# #640-recorded compatibility patches applied (gate_forward block_type
# dispatch, dataset loaders).
#
# EVERY bounded-cell edit vs that lane, recorded here:
#   MODEL             Qwen/Qwen3.5-2B -> local pinned Qwen3.5-0.8B snapshot
#                     (revision 2fc06364715b967f1860aea9cf38778875588b17)
#   NUM_GPUS          8 -> 1  (single RTX 3090, power-capped 280 W)
#   GRAD_ACCUM        1 -> 4  (effective batch 4; the lane ran 8x1 on 8 GPUs)
#   EPOCHS            20 -> 2  (bounded window; ~512 optimizer steps)
#   WARMUP_STEPS      1000 -> 50
#   article window    (kept lane-verbatim 2048..16384 — the generator's train
#                     bodies are hardwired ~U[14000,16350] tokens, so a lower
#                     cap only DROPS rows: the window filters, it does not
#                     crop. First attempt at 8192 filtered all 1024 rows.)
#   EVAL              freq 600->300, samples 100->25/group,
#                     eval article cap 39000->20000 (no filtering at 0.8B data)
#   MAX_GEN_TOKENS    1024 -> 32  (NIAH answers are short)
#   GRAD_CHECKPOINT   0 -> 1  (24 GiB)
#   PYTHON            -> the #640 venv explicitly
#   E2E_GATE_OUTPUT_ROOT -> /home/damen/642-runs
#   train-step env    set explicitly (TRAIN_ATTN_BACKENDS=flex, PACK_PHASE2=0,
#                     COMPILE_PHASE2=0) — verbatim what the driver's
#                     Qwen/Qwen3.5-* branch resolves to; the local pinned path
#                     does not match the glob. Packed store built for the 0.8B
#                     tokenization (python -m tokengate.datasets.pack iso_niah
#                     --model Qwen3.5-0.8B-r2fc0636 --root datasets).
# Everything else is the lane verbatim: iso_niah manifest, eval split
# longbench_eval_split_niah_iso_v1.json, LoRA r=16 alpha=32 on q_proj+v_proj,
# Muon + aux AdamW (MUON_GATE/NO_DECAY_SCALARS), LogUniform phase-2 budget
# sampling + full-context sentinel, SWA on, K eval 2000, gate layer 1 (DeltaNet),
# bias on the 6 full-attention layers above.
set -euo pipefail

MODEL=/home/damen/models/Qwen3.5-0.8B-r2fc0636
: "${TORCHRUN_MASTER_PORT:=29942}"
export MODEL TORCHRUN_MASTER_PORT
cd "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)"

PYTHON=/home/damen/tokengate-640/.venv/bin/python
E2E_GATE_OUTPUT_ROOT=/home/damen/642-runs
export PYTHON E2E_GATE_OUTPUT_ROOT
# The driver's Qwen/Qwen3.5-* kernel-default branch, verbatim (local path
# misses the glob). Packed store required by the target-balanced map.
TRAIN_ATTN_BACKENDS=flex
PACK_PHASE2=0
COMPILE_PHASE2=0
export TRAIN_ATTN_BACKENDS PACK_PHASE2 COMPILE_PHASE2

# ── Data ──────────────────────────────────────────────────────────────────────
PRETRAIN_MANIFEST=iso_niah
TARGET_BALANCED_MAP="tokengate/datasets/maps/eval_target_train_map_v17_niah_only.yaml"
: "${MIN_ARTICLE_TOKENS:=2048}"
: "${MAX_ARTICLE_TOKENS:=16384}"
EVAL_MIN_ARTICLE_TOKENS=0
EVAL_MAX_ARTICLE_TOKENS=20000
SHUFFLE_EPOCHS=1
PHASE2_LOGUNIFORM=1

# ── Schedule (bounded) ───────────────────────────────────────────────────────
LR_VALUES=1e-4
EPOCHS=2
: "${EVAL_FREQ:=300}"
: "${EVAL_SAMPLES_PER_GROUP:=25}"
SWA_FREQ=1
SWA_START=0
SWA_EVAL_FREQ=1
: "${GATE_FULL_EVAL_FREQ:=5}"
: "${GRAD_CHECKPOINT:=1}"
export GRAD_CHECKPOINT
: "${WARMUP_STEPS:=50}"
: "${MAX_GEN_TOKENS:=32}"

# ── Adapter + optimizer (recipe verbatim) ────────────────────────────────────
LORA_RANK=16
LORA_ALPHA=32
LORA_TARGETS="q_proj v_proj"
LORA_LR=1e-4
MUON_GATE=1
NO_DECAY_SCALARS=1

# ── Batch / distribution ─────────────────────────────────────────────────────
: "${TRAIN_BATCH_SIZE:=1}"
: "${NUM_GPUS:=1}"
: "${GRAD_ACCUM:=4}"

# ── Eval split / naming ──────────────────────────────────────────────────────
K_EVAL_VALUES="2000"
EVAL_SPLIT_MANIFEST="tokengate/datasets/eval_splits/longbench_eval_split_niah_iso_v1.json"
EVAL_GROUP_OVERRIDE="niah:niah_kamradt"
EVAL_TEST_FULL_TASKS="niah_kamradt:niah_heldout"
MULTITASK_SUFFIX="-642-bounded-08b-l1d"
GATE_LAYERS="1"
export GATE_LAYERS
export WANDB_TAGS="642,bounded,0.8b,qwen3p5,hybrid,l1d"
CHECKPOINT_KEEP_ALL_AFTER_WARMUP=1

export PRETRAIN_MANIFEST LORA_RANK
export LORA_ALPHA LORA_TARGETS LORA_LR
export GATE_FULL_EVAL_FREQ TRAIN_BATCH_SIZE
export MUON_GATE NO_DECAY_SCALARS PHASE2_LOGUNIFORM CHECKPOINT_KEEP_ALL_AFTER_WARMUP
export EVAL_SPLIT_MANIFEST EVAL_GROUP_OVERRIDE EVAL_TEST_FULL_TASKS
export MIN_ARTICLE_TOKENS MAX_ARTICLE_TOKENS EVAL_MIN_ARTICLE_TOKENS EVAL_MAX_ARTICLE_TOKENS
export SHUFFLE_EPOCHS
export LR_VALUES EPOCHS EVAL_FREQ SWA_FREQ SWA_START TARGET_BALANCED_MAP
export K_EVAL_VALUES SWA_EVAL_FREQ NUM_GPUS MULTITASK_SUFFIX
export GRAD_ACCUM WARMUP_STEPS MAX_GEN_TOKENS EVAL_SAMPLES_PER_GROUP

echo "[642] data: niah_kamradt/niah_kamradt — 1024 train / 200 test rows, 0.8B pinned tokenizer"
echo "[642] eval: every ${EVAL_FREQ} steps, ${EVAL_SAMPLES_PER_GROUP} samples/group, K=${K_EVAL_VALUES}, gate layer ${GATE_LAYERS}"
echo "[642] model: ${MODEL} (revision 2fc0636...) on ${NUM_GPUS} GPU, eff. batch ${TRAIN_BATCH_SIZE}x${GRAD_ACCUM}"

bash train-scripts/run_e2e_gate_pretrain.sh
