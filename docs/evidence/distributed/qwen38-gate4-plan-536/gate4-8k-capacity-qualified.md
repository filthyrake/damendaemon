# Gate 4 — 27B Q4 three-device capacity table and the 8K cell (#536 cycle 10)

Date: 2026-10-20 (UTC)
Runs: `scripts/experiments/gate4_27b_capacity.py` (planner math, CPU) and
`scripts/experiments/gate4_8k_attempt.py` (guarded live 8K attempt,
torchrun 3-rank A30 + A2 + desktop RTX 3090).

## Capacity table — choose_stage_splits planner, batch 1, budgets 22528/13312/22528 MiB

| context | storage plan | stage 0 (A30) | stage 1 (A2) | stage 2 (3090, head) | total |
|---|---|---|---|---|---|
| 2048 | found | 0–24, 7.83 GiB | 24–44, 4.65 GiB | 44–64, 7.68 GiB | 20.16 GiB |
| 4096 | found | 0–24, 7.96 GiB | 24–43, 4.56 GiB | 43–64, 8.01 GiB | 20.53 GiB |
| 8192 | found | 0–24, 8.21 GiB | 24–43, 4.78 GiB | 43–64, 8.26 GiB | 21.25 GiB |

Numbers are the planner's required-memory estimate per stage (packed Q4
weights + FP32 KV/conv/state + workspace). Every context fits its budgets
with wide margin — the two-rank pair's gate-3 problem (a ~21 GiB total
cell across 40 GiB with a growing allocator gap) is distributed across
64 GiB of budget here.

## Live 8K attempt — result: QUALIFIED

Context 8192, batch 1, chunked KV ramp (15 × 512-token `prefill_chunk`
rounds to 7680 tokens — a whole-prompt 8K prefill is never issued), then
12 greedy decodes under the standard autoregressive protocol; then
release + an identical second ramp as the determinism gate:

- **Deterministic replay:** ramp 2's final chunk sample == ramp 1's
  (1000), and both greedy sequences identical `[1001..1012]`;
  `deterministic_replay: true`, `failed: false`, all ranks exited
  cleanly. (Self-consistency is the honest gate available without a
  thermally risky 8K two-rank pair reference; byte-equality to a pair
  reference at 8K is not claimed.)
- **Measured per-stage peak memory** (torch max_memory_allocated):
  rank 0 (A30, layers 0–24) **8.26 GiB** vs 8.21 planned; rank 1 (A2,
  24–43) **4.75 GiB** vs 4.78 planned; rank 2 (3090, 43–64 + head)
  **5.94 GiB** vs 8.26 planned (the head-stage plan is conservative).
  All within the 22528/13312/22528 MiB budgets; no rank OOMed (any rank
  failure would have poisoned the epoch) — the pair's observed-vs-planner
  allocator gap is negligible at three-way splits.
- **Thermals:** no trip. External `nvidia-smi` polls during the run
  stayed ≤ 49 C on the A2 (the gate-3 65 C wall was a two-rank sustained
  ramp artifact; the A2's 8K tail moved to the 3090 as planned).
- **Ramp cost:** 15 chunks, ≈ 4.75–5.14 s/chunk steady (first chunks
  include Triton compile); 10 MiB chunk boundary hop ≈ 35.6 ms
  (cycle-9 microbench) — small next to chunk compute.
- **Decode at 8K context:** ≈ 270 ms/step steady (vs ≈ 137.6 ms at 2048 —
  growth tracks the wider KV/conv state; warmup-inclusive first step
  15.9 s is Triton compile, excluded from steady).

## Scope statements (honest)

- The 8K cell is qualified on the **runner path** (probe admission, batch
  1, self-consistency + clean completion + measured memory), not yet via
  `chain-serve` HTTP at 8192; the production `chain-serve` context cell
  remains 2048 (the policy `CHAIN_CONTEXT_LIMIT`). Raising the served
  cell to 8192 is a follow-up admission change once a served 8K pass
  (health + completions + chunked-prefill equality at 8192) is recorded.
- No throughput/SLO claim. Thermal polling was external samples, not a
  continuous trace; the in-process guard (rank-0 temperature) was
  unavailable on this driver and is recorded as such.

Evidence: `gate4-27b-capacity.json` (planner table),
`gate4-8k-attempt.json` (live run), `gate4-8k-rank{1,2}.json`
(worker-side measured peaks).
