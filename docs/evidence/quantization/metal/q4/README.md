# Native Metal Q4 qualification

Tracked in GitHub issues #4 and #23. Normal runtime admission now supports Qwen2.5-0.5B geometry,
standard RoPE, context through 2048 and one to four request slots.

The native shader reads Q4_0 low/high nibbles directly for one to four rows,
using FP16 scales and FP32 accumulation. Larger prompt matrices temporarily
expand one projection for the MPS matrix product. No persistent dequantized
model copy is introduced. FP16 activations round coefficients and outputs to
FP16; FP32 activations preserve FP32 range and coefficient products.

All 28 isolated matrix checks pass against independent CPU FP64 unpacking and
products. Cases cover signed/zero scales, every nibble value, output tails,
bias, row counts 1–5/17/31, and both activation precisions. This is numerical
qualification, not a performance result (`q4-metal-kernels.json`).

The laptop's context-512 prepared Q4 identity matches the pinned NVIDIA
Qwen2.5-0.5B checkpoint exactly:
`5344521d357deea794ab9fd3d06a6cdae292c3c0588d933899dc3e6139ff5ffe`.
The source revision is `7ae557604adf67be50417f59c2c2f167def9a775`.

An independent CPU FP32/gguf-Q4 reference covers 2048 context, long prompts,
reordered batched decode and replaced slots. The isolated Metal experiment
passes all 82 original gates, including exact greedy tokens. Worst relative
RMS is 0.000011355 and maximum absolute error is 0.000146270. Activations,
residuals and KV remain FP32, with FP16 retained unquantized matrix storage.
The wrapper records actual Q4 dispatch counts and restores public admission
after execution. Reports are `qwen2-q4-metal-{reference,replay,dispatch}.json`.

Reproduce the isolated replay using `scripts/experiments/run_metal_q4.py`
around `scripts/replay_reference.py`, with explicit `DLLM_DEVICE=mps`,
`DLLM_QUANTIZATION=q4_0` and `DLLM_METAL_WEIGHT_DTYPE=float16`. The normal runtime independently repeats all 82 model checks with identical
results (`qwen2-q4-metal-production.json`). Prefix-cache verification passes
48 checks covering exact/divergent prefixes, eviction and chunked prefill
(`qwen2-q4-metal-prefix.json`).

The real HTTP gate passes 19 checks: independent continuations, shrinking
concurrent batches, streaming, cancellation, invalid-input recovery, logprobs,
seeded sampling, Q4/Q8 switching, host restore and disk reload. The Q8 switch
is a lifecycle smoke check; Q4 outputs are compared with the CPU reference.
Repeated full unloads retain the same live Metal allocation count
(`qwen2-q4-metal-http.json`). Run `scripts/verify_metal_q4_service.py` against
the local CPU trace to reproduce. Performance measurements remain separate;
these correctness results do not establish a speedup.


## Initial serving measurements

M4, the same pinned checkpoint, 16 input IDs, context 512, 32 output tokens,
FP16 retained unquantized storage and FP32 activations/KV. Q4 and Q8 ran
sequentially, each with one warmup and five measured rounds. These are
**different precision** measurements, not an equivalent-arithmetic speedup.
No upstream engine was remeasured for this comparison.

| Precision | Concurrency | Aggregate tokens/s | First text ms | Request latency ms |
|---|---:|---:|---:|---:|
| Q4 | 1 | 47.76 | 47.86 | 669.96 |
| Q8 | 1 | 49.44 | 48.51 | 647.16 |
| Q4 | 2 | 83.47 | 80.96 | 757.51 |
| Q8 | 2 | 85.20 | 82.82 | 748.71 |
| Q4 | 4 | 148.51 | 122.98 | 861.35 |
| Q8 | 4 | 142.14 | 129.43 | 896.88 |

Values are medians of measured rounds or requests. There is no established
Q4 speed advantage: small differences here need counterbalanced repetition.
Raw reports are `q4-http-c*.json` and `q8-http-c*.json`.

A separate raw-prompt Q4 decode diagnostic records roughly 1.6 ms dispatch
and 18 ms token readback per step (`q4-host-profile.json`). Readback includes
waiting for preceding GPU execution; it is not transfer cost alone. Host API
time likewise can contain GPU waits. This does not justify attributing the
runtime gap to Python or predict a gain from a language rewrite.


The isolated SIMD-group packing experiment (`q4-metal-groups.json`) checks
64/128/256/512 threads per group with exact output equality, repeated
counterbalanced order, and Qwen2 projection geometries at one/four rows.
Smaller groups help the narrow output/down projections slightly, but do not
materially improve the large vocabulary projection (roughly 4.55 ms at one
row). Runtime launch defaults remain unchanged. This is a component benchmark,
not a measured end-to-end optimization.


## Guarded 32-bit indexing optimization

The projection's 64-bit inner-loop indices are expensive on M4. The runtime
now uses unsigned 32-bit indices when input, output and logical weight element
counts are below 2^32; larger layouts retain the original wide-index shader.
FP32/FP16 coefficient and accumulation behavior is unchanged. All 28 isolated
FP64-reference checks and all 82 independent model checks pass again.

The isolated projection experiment checks bit-identical outputs and repeats
baseline/candidate/candidate/baseline three times per shape. At the 151936 by
896 vocabulary matrix, one-row latency falls from 4.56 to 0.92 ms. Other
projection shapes also improve; see `q4-metal-indices.json`.

A separate full-model experiment alternates old/new/new/old three times on the
same resident weights, M4, Q4 precision, context 512 and 16 input IDs. Each run
produces the same 32 tokens exactly. Timing includes token readback for the 31
decode steps after prefill; it excludes prefill, startup and HTTP overhead.
Each mode has six measured runs after warmup.

| Requests | Old decode ms | New decode ms | Old aggregate tokens/s | New aggregate tokens/s |
|---|---:|---:|---:|---:|
| 1 | 625.67 | 249.78 | 49.55 | 124.11 |
| 4 | 750.73 | 445.60 | 165.18 | 278.29 |

These are matched optimization measurements: 2.50x single-request and 1.68x
four-request decode throughput. They are not whole-request or upstream-engine
comparisons. Reproduce with `scripts/experiments/bench_metal_q4_indices.py`;
reports are `q4-metal-index32-decode-c{1,4}.json`.

The optimized runtime also repeats all 19 real HTTP lifecycle checks, including
concurrent shrinking batches, Q4/Q8 switching and zero live Metal allocations
after both complete unloads (`qwen2-q4-metal-index32-http.json`).
