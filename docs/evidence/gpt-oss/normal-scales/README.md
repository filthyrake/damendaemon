# Validated normal-scale MXFP4 decode

At immutable checkpoint activation, one min/max reduction classifies each expert
projection's scale tensor. When every scale lies in [2,252], all nonzero E2M1
values become finite normal FP32 values, so their exact bits can be constructed
without generic subnormal/overflow handling. Signed zero is preserved. Any
out-of-range scale selects the existing generic projection. The separate exact
FP16-expansion range remains [104,140]. No validation scan occurs during replay.

## Correctness

- All 4,016 code/normal-scale combinations match an independent CPU FP64 E2M1
  table/ldexp reference bit-for-bit, including both signs of zero.
- Shared/per-route projections match the generic kernel exactly. Captured graphs
  follow changed expert IDs; negative/out-of-range/int64-wrap routes remain
  bounded and produce NaN rather than reading outside the expert bank.
- Generic all-256-scale unpack, FP16/BF16/FP32 conversion, compensated products,
  routed projections, activation and graph-bound tests still pass.
- All 72 independent GPT-OSS reference checks pass, including scalar graph
  decode and slot isolation/reuse. All 48 checkpoint expert projections qualify.
- 238 local CPU/HTTP tests pass, including whole-tensor scale-range selection.

## Matched production-path HTTP results on A2

Same pinned GPT-OSS-20B checkpoint, native MXFP4, FP32 residual/KV/output, grouped
prefill, chunked reductions, context 256/two slots and 32 output tokens. Four
alternating warmed pairs use separate CUDA graphs sharing weights/KV. Responses
and usage match exactly. These exclude model loading/first-use compilation.

| Prompt tokens | Concurrent requests | Generic wall seconds | Normal-scale wall seconds |
| ---: | ---: | ---: | ---: |
| 5 | 1 | 2.157 | 2.104 |
| 32 | 1 | 2.644 | 2.587 |
| 129 | 1 | 2.852 | 2.819 |
| 32 | 2 | 4.828 | 4.744 |

Single-request median decode rates improve from 16.22–16.29 to 16.58–16.67
tokens/s, about 2.2–2.4%. Complete HTTP wall time improves 1.2–2.4% in this run.
This is a small gain; the exploratory run had mixed whole-request latency at
129 prompt tokens, so do not generalize the exact percentages.

Reproduce with `benchmark_gpt_oss_decode.py --optimization normal-scales`.
Full production reports are the `mxfp4-normal-{primitive,fallback,reference,http}`
files in this directory. The initial isolated expert-projection experiment is
also retained; it is not the source of the HTTP claim.

## Rejected Tensor Core layout

A separate experiment padded each selected route to a 16-row matrix tile and
unpacked weights inside compensated high/low FP16 dot products. Relative RMS
error was about 3e-7, but the tested layout was 6.7–8.2× slower on the A2. It was
rejected and is not a runtime option. Its source and reproduction notes live in
`scripts/experiments/`; the full measurement report is retained here. This result
applies to that specific layout, not every possible Tensor Core MoE algorithm.
