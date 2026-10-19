# #536 gate-4 planning precheck: three-device A30+A2+3090 Q4 split

CPU planning only (`plan_q4_split.py --device2-mib`, no device allocation, no
execution, no performance claim), against the pinned Qwen3.8-27B `1d4bf0f2`
checkpoint and the guard-era per-device budgets (A30 22,528 MiB / A2 13,312 MiB
/ 3090 22,528 MiB). Reports: `dd536-gate4-plan-{2048,8192,32768}.json`.

## Result: a balanced three-stage plan exists at every bounded context

| context | stage 0 (A30) | stage 1 (A2) | stage 2 (3090) | max util |
| --- | --- | --- | --- | --- |
| 2048 | layers 0–24, 8,021 MiB (36%) | 24–44, 4,762 MiB (36%) | 44–64, 7,869 MiB (36%) | 36% |
| 8192 | layers 0–24, 8,405 MiB (37%) | 24–43, 4,891 MiB (37%) | 43–64, 8,461 MiB (37%) | 37% |
| 32768 | layers 0–25, 10,150 MiB (45%) | 25–43, 6,026 MiB (45%) | 43–64, 9,997 MiB (44%) | 45% |

Packed Q4 weights 15.8 GiB; the planner balances storage, not transfer time.

## Why this attacks the gate-3 thermal finding

The three-device plan cuts the A2's stage nearly in half versus the two-rank
8K plan (layers 24–43 = 19 layers / 4,891 MiB vs layers 0–46 tail = 18 layers
on 7,812 MiB plan — and, critically, the A2 no longer owns the embedding +
first-24-layer compute block). Gate-3 attempts 1–8 showed the A2 heats on
cumulative ramp energy; halving its per-chunk compute share is the structural
relief a duty cycle could not buy (707→1133 s stretched the same total energy).

## What this does NOT establish

- No cross-host collectives qualified: the 3090 is a second host (10GbE);
  per-token host-transfer cost is unmeasured until a bounded three-rank run.
- No execution: the two-rank in-process pipeline (`build_cuda_pipeline`) is
  single-host only; gate 4 needs the torchrun/NCCL path qualified for three
  ranks with the #514/#515 rendezvous fixes.
- 32K fits at 45% utilization on paper, but the gate-4 contract bounds the
  first qualification to a small context with measured transfer cost.

## Next bounded step (gate 4, cycle 1)

Three-rank qualification runner in the `hybrid_two_rank_gate.py` shape:
stage 0 → A30 (labnode3), stage 1 → A2 (labnode3), stage 2 → 3090 (desktop,
separate host), bounded context (start at 2K), exact greedy vs a single-rank
reference, and a measured per-token stage-boundary transfer cost reported
separately from single-host numbers — never claimed as an SLO.
