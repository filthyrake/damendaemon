# Qwen3.5 still-image qualification

Tracked in GitHub issue #50. Experimental still-image HTTP serving is now
opt-in on NVIDIA A2 for Qwen3.5-0.8B, with FP16 or Q8 text weights. See the
final section for current scope and lifecycle evidence. Earlier sections record
the qualification sequence, including unresolved intermediate numerical limits.

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

The expanded six-prompt run adds identical text with inverted image pixels,
reversed image order, and replacement of an image-bearing slot by a text-only
request. All 54 independent comparisons pass, including reordered batched
decode: worst relative RMS 0.00635574, maximum absolute 0.0787911, and every
greedy choice agrees. With identical tokens, changed pixels produce a maximum
logit difference of 0.845695 natively versus 0.838867 independently. This tests
executor state reuse, not a serving prefix cache (none is enabled here).

## Experimental CUDA graphs and warm timing

Graph-enabled decoding passes the same 54 independent checks, including changing
image offsets and replacing an image request with text. The experiment remains
opt-in and outside the normal runtime registry. All 357 local tests pass.

An A2 ABBA comparison uses identical image fixtures, prompt IDs, checkpoint,
precision, and fixed 32-token output. Every generated ID agrees between phases.
Each phase excludes one warmup and retains three measured trials. Pooled medians:

| Image input | Uncaptured total | Graph total | Uncaptured decode (31 tokens) | Graph decode |
|---|---:|---:|---:|---:|
| 256x256 square | 1.1095 s | 0.4015 s | 1.0505 s | 0.3408 s |
| Resized image | 1.1178 s | 0.4092 s | 1.0561 s | 0.3405 s |
| Two images | 1.1442 s | 0.4410 s | 1.0396 s | 0.3401 s |

These are warm engine timings, including RGB preprocessing and device copy but
excluding file decoding, HTTP, model loading, and graph capture. Synchronization
separates the vision and text stages. They demonstrate a roughly threefold
decode improvement over this experimental uncaptured path, not an advantage over
another runtime. First-token medians are 58–102 ms uncaptured and 61–101 ms with
graphs; graph replay targets decode. Raw trials and per-stage measurements are
retained under `evidence/vision/qwen35-images-a2-warm*` and the ABBA summary.

## Inline image input helper

`vision_messages.decode_image_messages` accepts user content lists containing
text and inline PNG/JPEG data URLs with automatic detail. The optional `vision`
dependency installs the tested Pillow 12.3.0. It honors EXIF orientation, converts
to RGB, preserves content order, and returns encoded-content SHA256 identities
for future cache policy. It fetches no remote URLs and opens no caller paths.

Defaults cap requests at four images, 8 MiB decoded encoded-file bytes, and
16 million decoded pixels across all images. Combined pixel dimensions are
checked before loading image pixels. Animation and MIME/format mismatches are
rejected. Six tests cover order, identity, source preservation, byte/count/pixel
limits, malformed input, unsupported roles/detail, JPEG, and text-only input.
The helper is not connected to HTTP handling yet; no image-serving support is
implied by installing the dependency.

## Q8 text with images

The experimental image path also passes all 54 independent model comparisons
with Q8_0 text projections and CUDA graphs. The reference uses upstream GGUF
0.19.0 quantization on text Linear weights only, retaining the established
separate vision policy. Worst relative RMS is 0.00437896 and maximum absolute
error 0.0643835; every greedy choice agrees. Changed-image and slot-reuse controls
also pass. This is on-load text quantization, not a claim of quantized vision
weights or arbitrary quantized multimodal checkpoint compatibility.

The image-aware execution class now lives in `dllm.vision_hybrid` and shares
`HybridExecutor.forward_embeddings` with the text-only path. Normal serving
dispatch remains unchanged. Both FP16 and Q8 variants pass all 54 independent
checks again with four request slots and graph replay after this consolidation.
The local suite now has 363 passing tests.

## Inline chat-to-model bridge

`VisionModelWeights` explicitly owns text and vision host weights and includes
both in its byte count. It decodes inline image messages, expands placeholders,
and returns an `ImagePrompt` with the full token count and prepared pixels.
The reusable executor owns the corresponding vision GPU weights. Text-only
requests retain the original text model's marker rejection behavior. Image
requests with tool calls remain rejected. Request validation can check content
without decoding pixels, leaving pixel decoding for the inference worker.

The complete inline PNG chat path passes all 54 Q8 independent comparisons again
with four slots and CUDA graphs. Expanded prompt token IDs exactly match the
independent tokenizer/processor construction. Worst relative RMS is 0.00437896.
All 364 local tests pass. This bridge is still explicitly constructed by the
verifier; normal model loading, server validation, and serving dispatch have not
been changed to enable it.

Next integration work belongs to issue #50: opt-in model/HTTP dispatch, accurate
residency and release checks, malformed-request recovery, streaming/cancellation,
and actual HTTP measurements. Keep the intermediate encoder limitations visible.

## Opt-in HTTP serving

`DLLM_EXPERIMENTAL_VISION=1` activates this backend for the qualified Qwen3.5
1024-wide, 24-layer geometry. Execution is restricted to NVIDIA A2, context at
most 1024, and at most four slots. Other text-model geometries retain their
normal execution. Install the `vision` extra for PNG/JPEG decoding. HTTP retains
its 1 MiB request limit; content validation does not decode pixels on HTTP
threads. Image decoding and expanded prompt accounting occur on the admitted
inference worker. Chunked prefill and prefix snapshots are explicitly rejected.

Live FP16 and Q8 service checks cover six independent image/text fixtures, exact
eight-token output and prompt usage, mixed concurrent requests, streaming output
and usage, client disconnect followed by a changed image, malformed base64 and
image-file recovery, alias switching, host restore, and complete unload/reload.
Text and vision host weights are jointly counted; status exposes vision host/GPU
weight bytes and actual CUDA allocated bytes. The pre-load estimate keeps vision
storage independent of text quantization and excludes unused MTP weights.

The FP16 lifecycle test exposed a hybrid warmup-stream issue: creating fresh
streams for each graph/model retained additional cuBLAS workspaces on reload.
Hybrid warmup now uses the existing serialized per-device stream helper. After
all model weights are unloaded, allocator traces identify only persistent
cuBLAS workspaces: 8,519,680 bytes for the Q8 run and 25,559,040 for FP16. Repeated
inference/unload cycles leave exactly the same allocation, with zero host-model
bytes and zero vision GPU weight bytes. The runtime does not clear private
cuBLAS state or invalidate live graph workspaces. Traces and raw HTTP checks are
retained under `evidence/vision/`; diagnostic tracing runs are not performance
benchmarks. Intermediate encoder absolute-error limitations remain unchanged.

Final ordinary-server runs (without allocation tracing) pass all 20 checks in
both precision modes; the local regression suite passes 366 tests. Reports are
`qwen35-vision-http-q8-final.json` and `qwen35-vision-http-f16-final.json`.
For the square image with eight output tokens, three warm Q8 HTTP requests took
119.8, 342.1, and 119.4 ms; FP16 took 138.2, 138.5, and 137.5 ms. The Q8 outlier
is retained and requires investigation. These short service checks include HTTP
and image decoding and are not a controlled cross-runtime performance comparison.
They must not be combined with the earlier 32-token core-only measurements.
