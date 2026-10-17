# Parallel Mamba-2 prefill experiment

`DAMENDAEMON_MAMBA2_SSD=1` opts into a block-SSD prefill kernel on CUDA Ampere or newer
with at least 99 KiB shared memory per block. It is selected only for prompts/
chunks of at least 128 tokens with the qualified 130M geometry (inner1536,
head64, state128). Short prompts, decode and other shapes retain recurrent scan.
The default remains disabled while further performance work continues.

The independent kernel computes causal interactions within 32-token blocks (initial evidence below used 16) and
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
output text agrees across all three engines and both damendaemon implementations.

| Engine / path | c1 tokens/s | c2 | c4 | First text ms c1 / c2 / c4 |
|---|---:|---:|---:|---|
| damendaemon serial recurrence | 52.06 | 54.51 | 58.40 | 495.27 / 744.75 / 1975.76 |
| damendaemon block SSD | 86.22 | 94.10 | 106.84 | 246.36 / 371.12 / 980.12 |
| llama.cpp | 90.54 | 100.95 | 117.39 | 172.45 / 365.07 / 759.25 |
| vLLM | 117.41 | 137.84 | 164.76 | 157.33 / 232.45 / 473.02 |

This halves damendaemon first-token latency but does not close the long-prompt gap.
The CUDA profile attributes about 117.6 ms/prompt to SSD and 103.6 ms to the
two main FP32 matrix-multiply shapes, out of 239.3 ms recorded device work.
Decode remains dominated by matrix-vector projection (3.13 of 3.53 ms/token).
Profiling overhead and the HTTP benchmark are distinct measurements.

## Rejected projection replacement

The standalone `scripts/probe_fp32_matmul.py` and `scripts/tune_fp32_matmul.py`
measure an independent TF32x3 matrix projection; it is not selected by runtime
code. The six FP64-oracle cases pass, including inputs above FP16 range. A sweep
of 64 tile/warp/stage configurations per main shape found no useful improvement:
1536x3336x768 best 3.52 ms versus cuBLAS 3.15 ms, and 1536x768x1536 best 1.59 ms
versus 1.50 ms. Counterbalanced initial measurements also favored cuBLAS. These
are isolated, repeated-matrix device tests, not additional serving benchmarks.
The measurements are retained under `projections/` to avoid repeating a failed
optimization or treating use of Tensor Cores as evidence of improved performance.


## 32-token block qualification

The selected SSD block is now 32 tokens. A 64-token experiment exceeded A2 shared
memory (163840 bytes required, 101376 available), so it is not used. The 32-token
kernel passes the same closed-form/random/state checks; both FP32 and Q8 pass 139
full-reference checks through 2048, plus chunk256/prefix and HTTP restoration.
Evidence is in `block32/`. No correctness threshold changed.

On the same pinned FP32, 1536-input, 32-output, context2048 serving workload,
c1/c2/c4 throughput is 90.99/99.54/113.62 tokens/s and median first text
232.19/349.66/921.05 ms. Every trial's text agrees with the 16-token run and
upstream comparisons above. This is another 5-6% throughput improvement; the
long-prompt vLLM gap remains. Independent GPU jobs were serialized on A2.

## Compensated projection experiment

`DAMENDAEMON_MAMBA2_COMPENSATED=1` adds an independent opt-in CUDA prefill projection
path for unquantized 130M geometry, at 128 or more packed prompt rows. It reuses
damendaemon's existing power-of-two-scaled high/low FP16 activation decomposition and
FP32 accumulation. Every selected checkpoint weight must be finite and exactly
representable as FP16, checked before GPU allocation; Q8 and unqualified
geometries/devices reject the option. Resident weights remain FP32 and each
projection temporarily converts its matrix. Decode retains its existing path.
There is no persistent extra weight cache. This is compensated approximate
arithmetic with explicit numerical qualification, not a lossy weight conversion.

The full 159-check reference trace passes through 2048, including four simultaneous
128-token prompts, reordered state and decode, with maximum relative RMS
0.00002930. Chunk/prefix and real HTTP/model restoration also pass. Evidence and
the matched benchmark are under `compensated/`; isolated projection measurements
include the cost of temporary weight conversion.

With both SSD and compensated projections enabled, the matched long-prompt
c1/c2/c4 throughput is 98.00/109.30/127.57 tokens/s and median first text is
197.64/391.34/783.32 ms. All output text matches previous runs and both upstream
comparators. Throughput improves across concurrency; c2 first-text latency does
not improve in this measurement. vLLM still leads this workload. Both switches
remain experimental and disabled by default.
