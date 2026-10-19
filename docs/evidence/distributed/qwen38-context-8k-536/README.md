# #536 gate 3 — context cap to 32K (policy) and the 8K two-rank device cell (not qualified)

Date: 2026-10-19. Policy merged in PR #544 (`e3807122`).

## Policy result

The hybrid pipeline's 2048 context cap was a pin, not a memory bound. The
gate-1 planner (`../qwen38-pipeline-plan-536/`) already showed the layer-46
split fits at 8K and FP16 KV at 32K still fits. The cap moved 2048 → 32768 in
`hybrid_pipeline_policy.validate`, `hybrid_pipeline_build.build_cuda_pipeline`,
and the serving-runtime admission; the checkpoint's `max_position_embeddings`
(262144 for the pinned 27B) binds above. Boundary test:
`test_pipeline_context_cap_is_32768`. Full suite 2284 passed.

## Planner refresh on the pinned checkpoint (CPU, `plan_q4_split.py`)

| context | stage 0 | stage 1 | notes |
| --- | --- | --- | --- |
| 8192 | 13,113 MiB (layers 0–46) | 7,812 MiB (46–64) | matches gate-1 plan |
| 32768 | 15,338 MiB (layers 0–47) | 9,043 MiB (47–64) | split moves to layer 47 |

Reports: `dd536-plan-8192.json`, `dd536-plan-32768.json`.

## 8K two-rank device cell: NOT qualified (five guarded attempts)

All attempts ran `scripts/experiments/hybrid_context_gate_536.py`
(--context 8192 --phase two-rank, near-limit trace 8160 tokens, 32 greedy
tokens) under `guard_shared_cuda` with the A30+A2 pair:

| attempt | config | guard stop | observation at stop |
| --- | --- | --- | --- |
| 1 | unchunked prefill, default reserve | 693 s, reserve gate | A30 free 2756 MiB |
| 2 | chunk 127, reserve 0 | 702 s, thermal | A2 64 C (gate < 65), A30 free 4782 MiB |
| 3 | chunk 127, reserve 0 | 692 s, thermal | A2 64 C, A30 free 4782 MiB |
| 4 | chunk 127, reserve 0 | 710 s, thermal | A2 64 C, A30 free 4782 MiB |
| 5 | chunk 127, reserve 0 | 680 s, thermal | A2 63 C, A30 free 4340 MiB |
| 6 | chunk 127, reserve 0, **`DAMENDAEMON_PIPELINE_EMPTY_CACHE=1`** | 707 s, thermal | A2 62→≥65 C, A2 free 897 MiB, A30 free 2642 MiB |
| 7 | attempt 6 + `DAMENDAEMON_PREFILL_CHUNK_COOLDOWN_S=3` | 961 s, thermal | A2 61 C at last pass then ≥65, A2 free 9857 MiB |
| 8 | attempt 6 + `DAMENDAEMON_PREFILL_CHUNK_COOLDOWN_S=6` | 1133 s, thermal | A2 62 C at last pass then ≥65, A2 free 9919 MiB |

Attempts 7–8 (PR #550) added a duty-cycled pause between non-final chunks.
Cooldown stretched the run (707 → 961 → 1133 s) and `empty_cache` held the
allocator footprint near plan (A2 ~5–8.8 GiB resident during the ramp vs
14.2 GiB without), but the A2 still crossed 65 C late in every ramp: the pair
heats on **cumulative ramp energy**, not instantaneous power, so no cooldown
inside the guard's 1800 s cap completes the 8K ramp. The 8K cell on the
A30+A2 pair is thermally infeasible under the containment contract; per the
issue's stopping condition the gate is re-scoped honestly: the qualified
production cap remains the gate-2 2K cell, and 8K moves to the gate-4
three-device topology (A30 + A2 + 3090) where the A2's stage tail can move
off the thermally-limited device. Evidence retained: `dd536-8k-c2-*`
(attempt 7), `dd536-8k-c3-*` (attempt 8).

`dd536-8k-guard.json` (attempt 5) and `dd536-8k.json` are retained.

### What the attempts establish

- Chunking at 127 removes attempt 1's reserve-gate failure mode (the #70/#535
  ramp bound works as expected), but the sustained chunked 8K prefill drives
  the A2 to the guard's thermal gate on every run. The #535 32K single-A30
  cell peaked at 53 C, so this is a two-device-pair thermal profile specific
  to this workload split.
- Observed per-device usage during the 8K request (A30 ~19.4–22.3 GiB,
  A2 ~11.1–14.2 GiB) runs well above the planner's required-memory numbers
  (13.1/7.8 GiB); the gap grows with context. The planner remains valid at
  the required-memory level, but the 8K allocator working set (caching
  allocator retention across 64 chunks plus attention working buffers) is not
  yet characterized.
- The served near-limit request never completed inside the thermal envelope:
  8K capacity on the pair is NOT device-qualified, and the qualified
  production cap remains the gate-2 2K cell.

### Path forward (recorded, not attempted)

- Bounded-duty prefill ramp (interleave cool-down between chunk groups) or a
  cooler ambient window before retrying.
- Characterize the allocator gap at 8K first (`torch.cuda.memory_summary`
  per stage after a synthetic ramp) — if retention dominates, `empty_cache`
  between chunk groups may pull the footprint back toward the plan.
- No gate relaxation: the thermal gate is a containment contract on shared
  hardware and stays as-is.
