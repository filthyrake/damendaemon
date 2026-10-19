# #536 gate 2 — two-rank Qwen3.8-27B Q4 on the labnode3 A30+A2 pair

Date: 2026-10-19. Gate-2 qualification of the rerouted two-device path from
gate 1 (PR #542): the hybrid CUDA pipeline (`build_cuda_pipeline`), not the
dense-only partition executor. Runs `scripts/experiments/hybrid_two_rank_gate.py`
under `scripts/experiments/guard_shared_cuda.py` with the two-device guard
(A30 UUID first, A2 second).

## Result: PASSED

- Two-rank serve (stage 0 on the A30 with a 22528 MiB budget, stage 1 on the
  A2 with a 13312 MiB budget): healthy, warmup + the pinned 64-token raw
  trace twice, `two_rank_completed: true`, both traces byte-identical
  (deterministic greedy).
- Single-rank A30 reference (same pinned checkpoint, Q4 compensated, no
  pipeline): greedy output identical to the two-rank output for the pinned
  trace — `greedy_matches_single_rank: true`, 12/12 completion tokens both.
- Per-device used memory after the traces (nvidia-smi): A30 13555 MiB,
  A2 4830 MiB — 59% and 37% of the stage budgets; after warmup
  12717 / 4704 MiB. Allocator totals from /v1/models/status: 17.68 GiB
  CUDA allocated across both devices.

Trace outputs (identical across ranks and phases):
`' same quantized weights.\n\nThe two-rank gate repl'` for the pinned
64-token seed, 12 greedy tokens.

## Scope and honesty

- Capacity and consistency qualification only. Shared guarded GPUs: no
  performance claim; the recorded `seconds` fields are request wall times
  under unknown incumbent load, not throughput or latency results.
- The issue text names a "single-rank CPU reference". A 27B FP32 CPU
  reference needs ~108 GiB of weights alone and exceeds the bounded window;
  the reference here is the single-rank A30 GPU path with the SAME Q4
  compensated quantization. Greedy agreement therefore qualifies the SPLIT
  (pipeline stage transport and coordinator cursor commit), not quantization
  drift — the Q4-vs-FP32 drift is covered separately by the #23/#539
  evidence at small scale.
- The first served request pays one-time Triton JIT compilation (measured
  ~141 s for the 16-token warmup in the pre-gate debug pass, and the
  single-rank reference request took 655 s including compilation; the same
  two-rank trace then completed in 17.7 s / repeat faster). This is the
  known #497 activation cost shape.
- The 2048-token context is the hybrid pipeline policy cap, not a memory
  limit; gate 3 (context/capacity admission) is a policy change with the
  planner headroom already recorded in `qwen38-pipeline-plan-536/`.

## Defects found and fixed by this gate

1. `hybrid.py` construction warmup ran the fused full-model probe
   (`forward_batch` → `forward_embeddings`) on layer stages, which
   deliberately block the full-backbone entry — stage construction failed on
   real CUDA at first request. Fixed: warmup restricted to full-model
   executors (`_layer_range is None`).
2. `hybrid_q4_policy.validate` pinned the device identity to the
   CURRENT device name, so stage 1 (A2) construction was refused. Fixed:
   under the explicit quantized-pipeline opt-in the identity anchors on the
   PRIMARY pipeline device (cuda:0 must be the A30); single-device admission
   unchanged.
3. `HybridPipeline._execute` moved stage tensors to the stage device but
   never entered that device's CUDA context; Triton launches run on the
   current device, so the A2 stage's kernels launched against device 0 and
   failed with `Pointer argument (at 0) cannot be accessed from Triton (cpu
   tensor?)`. Fixed: per-stage `torch.cuda.device(stage device)` in the
   coordinator, and the stage `output_logits` (fused Triton head) carries
   its own device context.

## Files

- `dd536-two-rank.json` / `dd536-two-rank-guard.json` /
  `dd536-two-rank.serve.log` — phase 1 (two-rank pipeline qualification).
- `dd536-single-rank.json` / `dd536-single-guard.json` /
  `dd536-single-rank.single.log` — phase 2 (single-rank reference + gate
  verdict). The final report carries the merged fields
  (`completed`, `greedy_matches_single_rank`, `two_rank_completed`).
- Pinned checkpoint: Qwen3.8-27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
  (the labnode3 HF-cache snapshot), Q4 compensated, context 2048, batch 1,
  no graphs, trace digest in the reports.
