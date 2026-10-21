# Qwen3-0.6B: the qualified FP32 path becomes the CUDA default (#2)

Date: 2026-10-21 (UTC)
Change: `damendaemon/qwen2.py` — `qwen3_default_precision(model)`: the CUDA
default for unpacked Qwen3 checkpoints is now the qualified FP32-arithmetic
path (`DAMENDAEMON_QWEN3_PRECISION` default `float32`); quantized weights
(packed kquant/GGUF scales, AWQ FP16 rounding) keep the `standard` fp16
default their own gates qualified. The env override still wins in both
directions.

All GPU runs guarded (`guard_shared_cuda`, A2 `GPU-1b29c7c8`), snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, torch 2.13.0+cu130.

## Acceptance gates

| Gate | Result |
|---|---|
| Stress replay vs independent FP32 reference (fresh trace, context 2048, 84 checks) | **84/84 passed, default env, no override** — worst relative RMS 3.28e-05 (0.0033%), greedy 84/84 (`replay-default-fp32.json`) |
| The historical failures under explicit `standard` opt-out | Reproduced exactly: event 8/slot 1 1.1932%, event 11/slot 0 1.4148%, greedy still correct (`replay-standard-optout.json`) — the opt-out is preserved, not broken |
| Batching (`verify_batch`) | batched prefill passed, 48/48 matched batched steps, slot reuse passed, max relative RMS 4.6e-06 (`batch.json`) |
| Model switching (`verify_model_service`, qwen2 + qwen3 + llama matrix, capacity 1 and 3) | passed; `residual_dtype: float32` visible in served timing metadata — the new default live through hot-swap (`model-service.json`) |
| Performance A/B (`compare_a2`, thinking off, 64 max tokens, medians) | see below |
| Qwen2.5-0.5B float32-reference cell (`verify_qwen2`) | 1.392% vs the 1% gate — **an unqualified configuration with no historical baseline, NOT a regression** (torch and code both exonerated; see corrected attribution below) |

## Performance tradeoff (recorded, unchanged decision boundary)

| conc | default (fp32) tok/s | explicit standard tok/s | fp32 TTFT ms | standard TTFT ms |
|---|---|---|---|---|
| 1 | 108.93 | 111.75 | 44.6 | 27.9 |
| 2 | 182.12 | 185.15 | 70.0 | 38.0 |
| 4 | 283.83 | 339.27 | 124.8 | 70.1 |

The default flip costs 1.6–2.5% at low concurrency and ~16% aggregate at
concurrency 4 (TTFT +17–55 ms) — the price of passing 84/84 instead of
failing 2/84 deterministically. Consistent with the historical compensated
Tensor Core measurements (107.23/170.68/271.50 vs standard 325.7 at c4).
Shared-GPU HTTP timings; not an SLO claim.

## Qwen2.5-0.5B float32-reference measurement — attribution CORRECTED (2026-10-21)

This section's original claim ("pre-existing torch 2.6-to-2.13 drift") is
**disproven** — see `../qwen2-gate-attribution-568/`. The bisect pinned torch
2.6.0+cu124 in an isolated venv and reproduced the 1.392% failure bit-for-bit
on the identical model/reference/code, while the historically-qualified
configuration (Qwen2.5-0.5B-Instruct + float16 reference, `regression-a.json`)
passes on the current torch (0.535/0.198/0.400% vs historical
0.598/0.187/0.508%). The 1.392% figure is a genuine measurement of an
unqualified configuration (base model vs a float32 reference) with no
historical baseline — not a regression from any code or environment change.
Greedy remains 16/16 on all three prompts in every run.

## Files

- `guard-*.json` — all seven guard reports (trace generation, both replays,
  batch, model service, both perf runs; all exit 0)
- `replay-default-fp32.json` / `replay-standard-optout.json` — the gate pair
- `batch.json`, `model-service.json` — batching + switching matrices
- `perf-{fp32,standard}-c{1,2,4}.json` — the A/B evidence
