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
