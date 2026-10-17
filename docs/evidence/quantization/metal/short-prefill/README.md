# Short-prompt packed Metal projections

For FP32 activations, keeping very short prompts packed avoids temporary
projection expansion. The bounded policy selects direct packed projection
through eight rows, or through sixteen rows for matrices with at least four
million logical coefficients. Larger prompt matrices keep MPS matrix products.
FP16 activations retain their existing four-row threshold. New direct-prompt
layouts must fit unsigned 32-bit element offsets; no persistent unpacked model
copy is added.

The component experiments use identical quantized coefficients and compare
packed accumulation with expanded MPS matrix multiplication. Floating-point
accumulation order can differ, so outputs are numerically checked, not claimed
bit-identical (`q4-metal-prefill-paths.json`, `q8-metal-prefill-paths.json`).
Larger rows can regress substantially; the policy deliberately stays bounded.

The full Qwen2.5-0.5B experiment alternates old/new/new/old three times after
warmup on the same M4, context 512 and input IDs. Timing includes prefill and
first-token readback, excludes HTTP and activation. Six samples per mode;
medians below. All output argmax IDs match; worst logit RMS difference is below
0.000003. Each row is a same-precision optimization comparison.

| Precision | Prompt tokens | Expanded ms | Bounded packed ms |
|---|---:|---:|---:|
| Q4 | 8 | 43.32 | 19.32 |
| Q4 | 16 | 42.31 | 33.57 |
| Q4 | 32 | 42.79 | 42.84 |
| Q8 | 8 | 46.62 | 18.57 |
| Q8 | 16 | 45.31 | 34.94 |
| Q8 | 32 | 45.44 | 45.35 |

Reproduce using `scripts/experiments/bench_metal_quant_prefill.py` with matching
`DAMENDAEMON_QUANTIZATION`, `DAMENDAEMON_DEVICE=mps`, `DAMENDAEMON_METAL_WEIGHT_DTYPE=float16`.
The harness explicitly forces the previous expansion threshold for its control,
so future default changes do not silently invalidate the comparison.

Independent CPU FP64 kernel checks now include eight rows and a large sixteen-row
matrix, in addition to signed/zero scales, tails, bias and FP16/FP32 activation
coverage. Both precisions pass all 36 checks with unchanged numerical tolerances.

Independent CPU FP32 model references pass every scalar prefix length 1–17,
mixed batches totaling 5/7/8/16/17 tokens, reordered decode, reused slots and
context boundaries. Qwen2 Q4/Q8 each pass 282 checks, Qwen3 Q8 passes 284 and
Llama Q8 passes 282. These reports use context 512 to focus on the changed
short-prompt path; prior 2048-context qualification remains separate evidence.

Qwen2 Q4/Q8 each pass 88 independent prefix-cache checks at context 1024. The
expanded checks restore a 255-token prefix and append 5/8/9/16/17 tokens,
covering direct packed and expanded offset-prefill paths. Existing divergent
prefix, eviction and slot-isolation gates remain unchanged.

Mamba and Mamba-2 Q8 each pass 159 CPU-reference checks, including explicit
eight-token batched prefill and recurrent-state checks. FP16-activation image
models retain their existing projection threshold.

Both Qwen2 precisions pass 59 real HTTP checks against independent short-prefix
traces: every selected boundary, concurrent shrinking batches, streaming,
cancellation, malformed-input recovery, sampling/logprobs, Q4/Q8 switching,
host restore, disk reload and repeated full release. Both full unloads return
live Metal allocation accounting to zero. Reproduce with
`scripts/verify_metal_quant_service.py` and the matching local CPU trace.
The regression suite passes 410 tests. No numerical tolerance was relaxed.

## Matched real HTTP measurements

Four fresh server runs in old/new/new/old order, each with one warmup and five
measured requests per prompt size. Ten measured requests per precision, prompt
size and mode; each mode uses identical input IDs, checkpoint, precision, M4,
context 512 and 32 generated tokens. Generated text and usage match exactly
between modes. These are medians; first text latency and whole-request latency
are reported separately from throughput. `expanded` is the previous policy;
`default` is the bounded policy.

| Precision | Prompt tokens | Policy | First text ms | Request ms | Tokens/s |
|---|---:|---|---:|---:|---:|
| Q4 | 8 | expanded | 44.03 | 294.26 | 108.73 |
| Q4 | 8 | default | 20.30 | 274.41 | 116.60 |
| Q4 | 16 | expanded | 43.16 | 295.28 | 108.36 |
| Q4 | 16 | default | 34.61 | 286.19 | 111.79 |
| Q8 | 8 | expanded | 46.15 | 364.30 | 87.83 |
| Q8 | 8 | default | 19.39 | 350.15 | 91.38 |
| Q8 | 16 | expanded | 45.23 | 368.04 | 86.94 |
| Q8 | 16 | default | 34.51 | 361.56 | 88.49 |

Reproduce using `scripts/experiments/bench_metal_prefill_service.py`. Startup and
activation are excluded by each server's warmup. This is a same-precision
optimization comparison, not an upstream-engine benchmark.
