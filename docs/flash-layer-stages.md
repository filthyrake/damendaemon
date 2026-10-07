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

First-stage input is nonempty CPU integer token IDs. Other stages accept the full
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
policy. Image-input remains a separate gate.

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
ranges and CPU reference scope. `--flash-images` with layer splits is rejected
before payload loading; this gate supplies only text/tool pipeline admission.

Tests check exact global admission, source receipts including final-construction
replacement, invalid request/cut preflight, native command equivalence and
actual queued HTTP subprocess JSON/SSE/drain/shutdown. Default and pipeline
responses agree. Stage cleanup failure remains observable and retryable. None
of this qualifies GPU execution, official checkpoints or practical performance.
