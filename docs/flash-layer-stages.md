# Packed Flash CPU layer stages

`glm_flash_reference_loader.load_reference_stage(snapshot, start, end, ...)`
audits the complete admitted Flash main-text checkpoint, then reads only the
selected contiguous range's payloads. The first range owns embeddings; the final
range owns final norm and vocabulary head. Visual and MTP values remain omitted.
Packed FP8 values stay packed and execute the existing CPU FP32 reference math.
Weight admission covers selected retained parameters plus the largest selected
source tensor for conversion overlap. Header/descriptor, execution/workspace and
future cache bounds remain separate.

`FlashLayerStage` is an explicit partial model, not a `FlashModel` subclass, so
existing whole-model generation and HTTP admission cannot accept it accidentally.
Every intermediate boundary retains `[batch, tokens, residual_streams, width]`
as finite CPU FP32 values. Only the final stage averages residual streams and
projects logits. Reducing streams at a layer cut would change the model.

First-stage input is nonempty CPU integer token IDs or explicitly prepared
finite CPU FP32 embeddings. Other stages accept the full
residual-stream chunk. Strict ranges, input geometry, vocabulary/cursors, finite
values, context and boundary output budgets are checked before mutation. The
existing mixed linear/sparse cache lifecycle provides per-range ownership,
aggregate cache reservation, slot reorder, partial-failure poisoning and retained
cleanup retry. Callers keep prepared weights immutable and own each returned
cache until close succeeds. This layer-stage API does not own a complete
pipeline, network, generator or image-input adapter.

Tests compose two/three ranges across different cuts, compare chunked/reordered
logits against independently expanded whole-model controls, inspect intermediate
stream shape and count selected packed payload reads. Exact selected weight
budgets, aggregate mixed-cache preflight, corruption outside the selected range,
invalid boundaries/cursors and partial allocation/cleanup failure are covered.
This is CPU fixture evidence, not official-model, GPU, practical-context or
performance qualification.

## Complete local pipelines and native generation

`FlashLayerPipeline` owns 2..8 distinct contiguous prepared ranges covering the
whole text model. Ranges must agree on configuration, dimensions and residual
stream count and cannot hide overlapping block objects. It coordinates one cache
cursor and slot order, admits the sum of fixed linear and capacity-dependent
sparse caches before allocation, and checks every residual boundary and final
logit budget before the first stage executes. Row growth is bounded by both the
aggregate cache allowance and the shared 128-row state limit.

Partial execution or reorder poisons all stage caches. Close attempts every
stage, retains failed owners, and can be retried. Failed allocation or prefill
cleanup is retried before another cache is admitted. GLM and Flash share these
ownership mechanics through `layer_pipeline.CpuLayerPipeline`; each family has
explicit stage type, cache-size and batch/activation-shape hooks. Neither family
pipeline inherits the other or masquerades as an ordinary complete model.

`FlashPipelineReferenceGenerator` and `FlashPipelineToolReferenceGenerator`
provide explicit native single/equal/ragged text and tool generation. They reuse
the pinned Flash format and existing bounded generation lifecycle; tool calls
are returned, never executed. Ordinary model admission remains unchanged. HTTP explicitly admits the native
Flash pipeline tool adapter, with the same reference authentication and request
policy. Separate image adapters admit prepared still-image features as described below.

Pipeline tests compare real packed chunked/reordered logits and native text/usage
against independent whole-model CPU controls, exercise aggregate budget and
retained failure ownership, and test cancellation/abandonment/reuse. Tool tests
select controlled tokens after real stage execution to verify protocol handling;
they do not establish trained-model tool quality.


## Checkpoint builder, native command and reference HTTP

`glm_flash_checkpoint_pipeline.load_reference_pipeline(snapshot, layer_splits,
...)` preflights all retained stage weights plus the largest conversion source
against one total allowance, before reading any tensor payload. It then loads
only each assigned range, checking config/shard/index receipts before and after
loads and after complete construction. These ordinary file-stat receipts detect
concurrent replacement/modification; they are not content authentication or a
claim against hostile writers. The same global-admission implementation serves
GLM through explicit family callbacks.

Use `--layer-splits` with `scripts.run_glm_flash_reference` to select the local
pipeline for native prompt/history, equal/ragged batches and tools. Cuts must be
ordered distinct internal layer indices; a three-layer fixture can use
`--layer-splits 1 2`. Defaults still select the whole-model reference. Output
identifies `execution: local_cpu_pipeline` and the complete stage ranges.

`scripts.serve_cpu_reference flash ... --layer-splits ...` admits the explicit
pipeline tool adapter to the existing loopback HTTP service. It supports the
same authenticated native JSON/SSE, optional serialized request queue and admin
drain as the whole-model reference. Readiness, status and timing expose pipeline
ranges and CPU reference scope. Add `--flash-images` for the explicit image/tool
pipeline adapter with a separately bounded vision model.

Tests check exact global admission, source receipts including final-construction
replacement, invalid request/cut preflight, native command equivalence and
actual queued HTTP subprocess JSON/SSE/drain/shutdown. Default and pipeline
responses agree. Stage cleanup failure remains observable and retryable. None
of this qualifies GPU execution, official checkpoints or practical performance.

## Still-image pipelines

`FlashLayerPipeline.forward_embeddings` and `prefill_embeddings` accept finite
CPU FP32 `[batch, tokens, width]` input at the first stage. Residual streams are
expanded exactly once, then preserved through every text-stage boundary. Normal
token decode and row reorder continue with the same mixed-cache owner. Invalid
feature geometry and all intermediate/final output budgets fail before execution.
Partial stages are never accepted by the image-feature insertion helper.

`FlashPipelineImageReferenceGenerator` and
`FlashPipelineImageToolReferenceGenerator` reuse the pinned image processor,
placeholder counts, native chat/tool format and cleanup lifecycle. Use
`--layer-splits` with `scripts.run_glm_flash_image_reference`, or combine it with
`--flash-images` in `scripts.serve_cpu_reference`. Text pipeline and vision weight
allowances remain separate; processor, image/workspace, context and cache limits
still apply. Queued HTTP retains still-image modality and full stage metadata.

Tests compare prepared-feature chunks, reordered state and subsequent token
decode against an independently expanded whole-model control. Real packed
vision/text fixtures exercise native commands, tools, concurrent queued HTTP,
JSON/SSE, drain and shutdown. Cancellation and retained cleanup block reuse until
release succeeds. Controlled tool tokens prove parser plumbing, not trained image
or tool quality; this remains local CPU evidence only.

## Independent request slots

`FlashStageSlots` reserves the complete mixed-cache cost for 1..4 independent
requests before any cache allocation. Linear state contributes fixed bytes per
slot; sparse state scales with the configured context. Each occupied slot owns
its own cache and cursor, allowing unequal prompt lengths and reordered decode.
Release closes only selected occupied slots and permits fresh reuse.

First-stage embeddings expand into full residual streams. `forward_layers`
accepts `[batch, tokens, streams, width]`, and even the last stage returns that
shape: only explicit `output_logits` reduces streams and applies the head.
`FlashLayerStage.forward_activations` exposes the same layers-only boundary.
Malformed inputs, positions, slot lists and boundary budgets fail before mutation.
A partial row/allocation, head or release failure retires the slot owner, attempts
every cleanup, and retains failed state for close retry.

GLM and Flash share lifecycle mechanics in `CpuStageSlots`, with separate explicit
stage type, configuration, activation shape, embedding and cache-cost hooks.
The internal base is not directly constructible. Host transports still admit
only their previously supported families; Flash transport integration is a
separate gate. Packed two/three-stage controls cover chunks, unequal cursors,
reordered decode, exact mixed-cache reservation, release/reuse and failures.

## Prepared host stages

`HostTopology` explicitly validates Flash root/text geometry and FP8 reference
precision. Its identity binds the complete configuration, residual stream count,
layer ranges, source/session policy, context, slots and mapped vocabulary. Wire
width is `hidden_size * hc_mult`; frame and chunk limits cover every stream.
Configured image/video/audio IDs remain excluded from this text-only path.

`HostPipelineCoordinator` and `HostPipelineWorker` admit explicit Flash slot
owners. They flatten streams only for bounded transport rows and restore the full
shape before each range executes. No stream reduction occurs between stages.
All participants admit each command before execution. Intermediate stages return
full activations; the last stage returns policy-checked sampled token IDs.
Partial execution, peer loss and abandoned completion discard the whole epoch.

`glm_flash_host_pipeline_build.build_glm_flash_host_pipeline_worker` claims
already authenticated, fresh, identity-matched transports, then loads only the
assigned packed range. Optional bounded snapshot fingerprinting checks actual
bytes and rechecks receipts across construction; without it, the source identity
is an operator assertion. Shared GLM/Flash construction uses explicit family
callbacks. Invalid controls leave links caller-owned, while post-claim failure
closes every resource and retains failed cleanup in the build error.

Ephemeral loopback TLS tests compare packed two/three-stage logits and tokens
with independent whole-model controls; they cover ragged/chunked prefill,
reordered decode, release/reuse, STOP, identity/frame rejection, peer failures,
range-local loads, source mutation and retryable construction cleanup. These prepared APIs do not qualify actual remote hardware, image-host execution
or speed. Native command and explicit HTTP admission are described below.


## Native host text and tool command

`HostFlashGenerator` binds the pinned Flash renderer/tool parser, a canonical
snapshot of the tokenizer, stop/media policy and verification mode to the host
session identity. GLM and Flash share `NativeHostGenerator` ownership mechanics
with explicit family hooks; their native identities and exact admission remain
separate. Single/equal/ragged conversations are prepared before dispatch. Invalid
input leaves a healthy epoch reusable; cancellation, abandonment, parser or peer
failure after dispatch discards it. Success finals follow every remote release
and parser cleanup. Media token IDs are excluded from this text-only path.

`scripts.run_glm_flash_host_topology` is the explicit multi-process command.
Supply `--native-chat` on every worker and coordinator, an ordered `--plan`,
coordinator `--peers`, pinned TLS material and the existing bounded resource
options. `complete` accepts one of `--prompt`, `--messages`, or
`--batch-messages`; tools, reasoning options and ragged batches use the same
pinned native format as the local Flash reference. Raw tokens are rejected. The `serve` role uses the explicit native host HTTP
adapter described below. The coordinator publishes final events
only after every peer acknowledges STOP. Optional `--verify-model-id` hashes the
bounded snapshot before connecting, then rechecks receipts after stage loading.

Tests use actual two/three-process loopback TLS workers and independent packed
CPU controls for native text/ragged/tool requests. Controlled tool tokens after
real stage math cover parser behavior, not trained tool quality. Invalid request
preflight, identity/family/media policy, failed STOP, cancellation and retained
cleanup are covered. Existing GLM and Qwen command behavior remains unchanged.


## Native host HTTP and serialized admission

`HostFlashReferenceRuntime` and `QueuedHostFlashReferenceRuntime` explicitly own
Flash native host runners. GLM/Flash runtimes share terminal-epoch and queue
lifetime mechanics through exact family bindings; the generic local runtime
still rejects host runners. Status exposes the Flash family, stage ranges,
source/topology IDs, verification mode and terminal state. This is text/tool HTTP
only; inline images remain limited to the local image pipeline.

Use `scripts.run_glm_flash_host_topology serve ... --native-chat` with the same
prepared worker plan, peer pins and resource bounds. Existing reference API and
admin authentication apply to native JSON/SSE, status and drain. Admission is
immediate by default; `--queue-size` opts into a bounded serialized queue with
input snapshots, client admission turns and deadlines. `--stop-after-drain`
requires a queue and acknowledges STOP only after outstanding work finishes.
PING uses the worker command lock and preserves occupied slots during output
backpressure. Idle signal shutdown can stop gracefully; active cancellation or
peer/parser failure discards the epoch and never reports successful completion.

Tests cover authenticated local TLS/HTTP and real subprocess JSON/SSE, native
tools, simultaneous queued requests, overflow/expiry, drain, signals, peer loss,
backpressure and retryable cleanup. Long queue-lifetime fixtures zero the
untrained vocabulary head before loading, keeping real packed stage execution
while emitting ordinary tokens. Independent numerical/native-format tests remain
separate. No GPU, trained-quality, remote-network or service-capacity claim follows.
