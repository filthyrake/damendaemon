# Qwen3.5 still-image qualification

Tracked in GitHub issue #50. This is an experimental standalone encoder, **not
multimodal serving support**. No normal model registry or execution path uses it.

The first implementation reads the vision tensors from Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`. Projections use FP16 storage with
FP32 compensated CUDA products; normalization, residuals, attention, and output
features use FP32. Vision weights occupy 204,977,152 bytes on host and device.
This precision policy is separate from text-model quantization.

`scripts/reference_qwen35_vision.py` generates independent CPU FP32 targets with
Transformers 5.18.0. It covers square, landscape, resized, larger, and paired
images. `scripts/verify_qwen35_vision.py` checks patch projection, learned
positions, each transformer block, and merged image features. The runtime
encoder imports no Transformers code. Published model equations and checkpoint
layout are the interoperability specification.

## Initial result: blocked

On NVIDIA A2, the first square image passes relative RMS error below 0.0001 at
every stage, but block 11 fails the separate absolute-error limit of 0.02:
maximum error 0.0578003. Merged features pass both limits (relative RMS
0.0000137586, maximum error 0.000140548). Neither ordinary FP32 projection nor
explicit eager attention resolves the intermediate failure. The verifier stops
at this first failing case; other native cases and timings are not qualified.
Limits remain unchanged. The independent oracle completed all five fixtures.

Raw reference metadata and negative controls are in `evidence/vision/`.
Source hashes identify the exact encoder revision used for each run. Diagnostic
attention/backend switches were added afterward; the default arithmetic is
unchanged. Existing runtime suite: 351 tests pass.

Remaining work includes numerical diagnosis, independent preprocessing,
image/text fusion and multimodal rotary positions, decode state, image-aware
prefix identity, serving input/lifecycle handling, and end-to-end performance.
Encoder timing alone will not establish multimodal request performance.
