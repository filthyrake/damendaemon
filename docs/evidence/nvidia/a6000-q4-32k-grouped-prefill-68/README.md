# A6000 Q4 32K grouped chunk-prefill A/B — OOM-retry storm eliminated (#68, 2026-11-01)

**Result: PASS — all gates, clean attribution.** The standing 32K prefill
OOM-retry storm (12–30 CUDACachingAllocator warnings per round in every
October/November cell) is attributed to the eager SDPA score block and
eliminated by dispatching the bounded grouped path on CUDA. Guarded run
exit 0, 701 s, both passes.

## Root cause (arithmetic-pinned)

The CUDA eager SDPA chunk path materializes a score block
`(1, 64 q-heads, 127 chunk, end)` fp32 growing linearly with the prefix.
The storm's last recorded allocation, 769,654,784 bytes, is exactly
23,680 KV positions × 64 heads × 127 tokens × 4 bytes — the score matrix
itself. The allocator rides the card ceiling and retries (~10 s per wave).

## Change (branch cuda-grouped-prefill)

`use_grouped_prefill` admits CUDA compensated chunk prefill (was
ROCm-only from the HOMER bring-up): the grouped kernel is pure-torch GEMM
— accelerator-agnostic, KV shared across query heads, fixed 256 MB
scratch — and the CUDA Q4 compensated surface exists since #679's
structural admission. `DAMENDAEMON_HYBRID_GROUPED_PREFILL=0` is the A/B
control override; everything else (precision, phase, shape bounds)
retained.

## Measured (pinned workload: Q4 compensated, 32K context, 32,256-token
prompt, 64 output, plain production serve — two sequential boots)

| | Pass A (grouped, default) | Pass B (SDPA control) |
|---|---|---|
| OOM warnings | **0** | **12** — storm reproduced |
| Allocator peak | 22.27 GiB | 24.78 GiB |
| First output | **300.8 s** | 327.3 s |
| Output digest | `8a700db8adf132dc…` | **identical** |

Gates: A completes within the C1 envelope (≤600 s) — 300.8 s; A's OOM
count strictly below B's — 0 < 12; B reproduces the storm (attribution
control) — 12 warnings. Cross-kernel greedy digests compared but not
gated per #23 — they came out **byte-identical** anyway.

## Tests

`tests/test_grouped_attention.py` — dispatch admits rocm+cuda compensated
chunk prefill and refuses every other precision/phase/shape; the A/B
override forces the SDPA path and rejects garbage values; the math-parity
tests (SDPA reference, multiple partitions, geometry/budget rejections)
unchanged and green.

## Artifacts

| File | sha256 |
|---|--- |
| ab-report.json | b76ba959fcb6852708e648be9bb28eff83c8d97c07c4df07ad454cec6068de16 |
| guard-ab.json | ff47b32e78abc38a58c8595c9e6441c6b50c42a9fb8e0b6dd64948ed9366a589 |
| passA-grouped.serve.log | d4c11fa932ea5315cb46ffdfdc2c31717bad5514da37bcf8976d3ee5cf2c3415 |
| passB-sdpa-control.serve.log | f361c64754724161bcaa30324a9d94315007a8c6ae94a85012162812751be315 |
