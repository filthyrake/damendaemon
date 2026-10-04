# Parallel Mamba-2 prefill experiment

`DLLM_MAMBA2_SSD=1` opts into a block-SSD prefill kernel on CUDA Ampere or newer
with at least 99 KiB shared memory per block. It is selected only for prompts/
chunks of at least 128 tokens with the qualified 130M geometry (inner1536,
head64, state128). Short prompts, decode and other shapes retain recurrent scan.
The default remains disabled while further performance work continues.

The independent kernel computes causal interactions within 16-token blocks and
carries FP32 state between blocks. Inputs, state and accumulation remain FP32;
matrix products use Triton's `tf32x3` decomposition. This is a declared arithmetic
change, not a claim of bitwise equality with the serial operation order. It does
not quantize activations or delegate to an upstream model runtime.

Qualification includes closed-form constant-input outputs/state, randomized
recurrent-vs-SSD comparisons, reordered slots and continued state. Both FP32 and
Q8 weights pass the full 139-check CPU-reference trace through 2048, plus state
and exact greedy generation. Chunk256 continuation/prefix checks and real HTTP
with chunk128, prefix snapshots, concurrent requests, cancellation and host
restoration also pass. The test scripts record whether SSD was enabled.

The initial prefix layout failed compiler/numerical checks and was replaced by
an explicit small triangular sum. Unsupported small/other shapes use the existing
recurrence; the qualified path passes the original strict gates. No threshold
was relaxed to admit it.

Native recurrence-stage medians for 1536 tokens, one row/reset: 15.32 ms serial
versus 4.91 ms SSD; these isolate one layer's recurrence, not full inference.
Serving comparison uses the pinned original FP32 checkpoint, exact same 1536
input IDs, context2048, 32 outputs and five warm rounds on A2. Every trial's
output text agrees across all three engines and both dllm implementations.

| Engine / path | c1 tokens/s | c2 | c4 | First text ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| dllm serial recurrence | 52.06 | 54.51 | 58.40 | 495.27 / 744.75 / 1975.76 |
| dllm block SSD | 86.22 | 94.10 | 106.84 | 246.36 / 371.12 / 980.12 |
| llama.cpp | 90.54 | 100.95 | 117.39 | 172.45 / 365.07 / 759.25 |
| vLLM | 117.41 | 137.84 | 164.76 | 157.33 / 232.45 / 473.02 |

This halves dllm first-token latency but does not close the long-prompt gap.
The CUDA profile attributes about 117.6 ms/prompt to SSD and 103.6 ms to the
two main FP32 matrix-multiply shapes, out of 239.3 ms recorded device work.
Decode remains dominated by matrix-vector projection (3.13 of 3.53 ms/token).
Profiling overhead and the HTTP benchmark are distinct measurements.
