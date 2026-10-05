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

Remaining work includes numerical diagnosis,
image/text fusion and multimodal rotary positions, decode state, image-aware
prefix identity, serving input/lifecycle handling, and end-to-end performance.
Encoder timing alone will not establish multimodal request performance.

## RGB preprocessing

`dllm.vision_inputs.prepare_images` matches the independent reference pixel
tensors exactly for all five fixtures on the production Torch 2.6 environment.
This includes resizing 320x192 to 352x224 and preserving paired-image ordering.
Byte-domain antialiased bicubic interpolation and fused normalization are
necessary for exact agreement; resizing in floating point and rounding only at
the end produces different pixels. The helper validates the combined patch
budget before resizing and leaves source tensors untouched. Three input/budget/
isolation tests pass. Encoded-image decoding and serving remain unimplemented.

## Expanded checks

All five native encoder cases pass the original final-feature limits: maximum
relative RMS 0.0000153751 and maximum absolute error 0.000217557. All stages
except block 11 pass both limits. Block 11 fails the absolute limit for every
case (0.0546–0.1057); its relative RMS still passes. No limits were changed and
no failed case was benchmarked.

An independent Transformers CPU-versus-CUDA control, with both matrix and
cuDNN TF32 disabled, also fails the same block-11 absolute limit for all five
cases (0.0371–0.0857). Its largest reference activations are about 2,400.
This establishes a limitation of the absolute intermediate check across
arithmetic implementations; it does not constitute end-to-end qualification.
The source-weight FP32 diagnostic also remains a failed check. Regenerate the
control with `reference_qwen35_vision.py --device cuda`, then compare artifacts
using `compare_vision_references.py`.

`vision_positions.py` independently matches prompt positions, decode offsets,
and rotary cosine/sine values exactly for seven layouts against Transformers
5.18.0, including text-only and two-image prompts plus eight subsequent decode
positions. Physical cache offsets remain distinct from rotary offsets. These
helpers are not yet wired into the hybrid executor.

## Experimental image/text computation

`scripts/experiments/vision_qwen35.py` combines native preprocessing and image
features with the hybrid text executor in a standalone validation path. It is
not registered for serving; graph capture is disabled. Image-specific rotary
offsets are stored per slot while KV addressing uses physical token positions.

An independent Transformers full-model run provides three image-conditioned
prompts (square, resized, and two images), each followed by eight teacher-forced
decode steps. Reference text weights/arithmetic use FP16; the reference vision
module is promoted from that loaded FP16 checkpoint to FP32 arithmetic. This is
distinct from the earlier source-BF16-to-FP32 encoder oracle. Native vision uses
FP16 projection storage and compensated FP32 arithmetic, with image features
cast to the text embedding precision during fusion.

All 27 full-model comparisons pass the existing logit gates (greedy agreement,
relative RMS below 0.01, maximum absolute below 0.5). Worst RMS is 0.00367180 and
maximum absolute error 0.0560737. Decode slots are reordered and batched. This
does not clear the separate intermediate encoder failures, prove semantic image
quality generally, or establish serving/performance support. Same-text/different-
image and slot-reuse controls remain next.
