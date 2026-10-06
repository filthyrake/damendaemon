# Shared Metal Q8 indexing qualification

The native packed Q8 projection uses 32-bit unsigned index arithmetic when
input, output and weight element counts fit below 2^32. Larger tensors keep
the original wide-index shader. Coefficients, accumulation and output rounding
are unchanged; prompt matrices above four rows retain their existing path.

Independent CPU FP64 projection checks cover FP16 and FP32 activations, signed
and zero scales, nonzero one-block weights, tails, bias and row counts through
31. All 28 pass (`q8-metal-kernels-index32.json`).

Full-model CPU reference gates pass without changing tolerances:

| Existing model path | Checks | Coverage |
|---|---:|---|
| Qwen2.5-0.5B Q8 | 82 | Context 2048, batched/reordered decode, replaced slots |
| Qwen3-0.6B Q8 | 84 | Context 2048, batched/reordered decode, replaced slots |
| Llama-3.2-1B Q8 | 82 | Context 2048, batched/reordered decode, replaced slots |
| Mamba-130M Q8 | 139 | Context 512, recurrent states, reused slots, exact generated tokens |
| Mamba2-130M Q8 | 139 | Context 512, recurrent states, reused slots, exact generated tokens |

The component experiment alternates old/new/new/old three times on identical
coefficients, comparing exact output bits for each shape. For the Qwen2
151936 by 896 vocabulary projection at one row, median time drops from 4.24
to 1.39 ms. This includes dispatch and completion; it is not full-model serving
latency (`q8-metal-indices.json`).

Qwen3.5-0.8B Q8 real image-chat HTTP passes all 20 existing checks, including
independent single/multiple/changed-image outputs, concurrent requests,
streaming, disconnect recovery, precision-preserving model swaps and repeated
release without live allocation growth (`qwen35-q8-metal-index32-http.json`).
These exercise the shared kernel's FP16 activation path. The local regression
suite passes 408 tests; no tolerance or admission expansion accompanies this
indexing optimization.

## Matched full-model decode

Qwen2.5-0.5B Q8, M4, context 512, the same 16 input IDs and 32 output tokens.
Timing covers 31 decode steps after prefill, including token readback; it
excludes activation, prefill and HTTP. Old/new/new/old repeats three times
after warmup, with identical outputs and identical host wrapper dispatch for
both modes. Six measured samples per mode; table values are medians.

| Requests | Old decode ms | New decode ms | Old aggregate tokens/s | New aggregate tokens/s |
|---|---:|---:|---:|---:|
| 1 | 614.92 | 324.57 | 50.41 | 95.51 |
| 4 | 736.94 | 401.25 | 168.26 | 309.03 |

This is 1.89x single-request and 1.84x four-request decode throughput on the
same precision and hardware. It is not an upstream-engine comparison or
whole-request latency claim. Reproduce with
`scripts/experiments/bench_metal_indices.py --quantization q8_0` and matching
`DLLM_QUANTIZATION=q8_0`, `DLLM_DEVICE=mps`, `DLLM_METAL_WEIGHT_DTYPE=float16`.
