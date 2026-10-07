# GLM CPU layer stages

`dllm.glm_reference_loader.load_reference_stage(path, start, end, ...)` loads an
explicit half-open GLM layer range from the same admitted local checkpoint as
`load_reference_model`. It audits the full checkpoint header inventory but reads
only assigned layer payloads, plus embeddings for the first stage or norm/head
for the final stage. FP8 projections and expert matrices stay packed. The weight
budget covers selected retained tensors plus the largest selected conversion
source; it does not require room for unselected layers.

Cuts must preserve sparse-indexer groups: the start and every non-final end must
be a full-indexer layer. A shared-indexer layer cannot begin a stage because it
needs selections computed by its preceding full layer. This initial contract
keeps that state within one stage. It does not transmit or recompute the missing
selections to permit arbitrary cuts.

`GlmLayerStage` also accepts caller-owned prepared blocks and boundary weights.
Keep these weights immutable. Call `new_state` for owned caches, then `forward`
with the exact current position. The first stage takes CPU int64 tokens; later
stages take finite CPU FP32 hidden chunks. Non-final stages return every hidden
row. The final stage returns last-token logits by default, or all logits with
`all_logits=True`. Existing model-state cache limits, row reorder and retryable
cleanup apply. Stage output limits include boundary tensors and final logits.

Each stage commits its own cursor after its successful result. An advanced
stage cannot be rolled back after another stage fails. A surrounding owner must
discard the entire pipeline epoch after partial execution or partial reorder;
this module does not claim distributed transactions or own a transport. Invalid
input detected before execution leaves the local cache reusable. Partial local
execution poisons that stage, and failed prefill cleanup remains owned for retry.

CPU tests compare prepared two/three-stage execution and separately loaded packed
checkpoint ranges against whole-model floating controls across chunked prefill,
decode and reordered/shrinking rows. They assert selected payload reads and exact
storage-budget admission. These are synthetic software controls. Host protocol,
bootstrap and serving integration, official checkpoint qualification, accelerator
execution, capacity and performance remain open in #51/#56.

## Owned local composition

`GlmLayerPipeline(stages)` validates 2–8 distinct stages covering the complete
model in order, with matching checkpoint configurations and boundary dimensions.
It exposes `new_state`, `prefill` and `forward`; its returned state supports
`reorder` and `close`. A single cache budget covers all stage reservations,
including growth through duplicated rows. Output-size admission runs before the
first stage executes.

The owner poisons every stage after a partial forward or reorder failure. Its
outer cursor advances only after the final result succeeds. Close attempts every
stage even if an earlier release fails and keeps unsuccessful releases for retry.
Failed allocation/prefill cleanup is retained by the pipeline or the allocating
stage and retried before another allocation. Invalid input rejected before work
leaves the state usable; an externally altered stage cursor poisons the epoch.

This is synchronous local CPU composition over caller-prepared stages, with no
transport, model registration or automatic serving admission. Packed checkpoint
controls compare logits with an independently expanded whole model. Fault tests
cover later-stage execution, reorder, allocation and cleanup failures. A fresh
state can be used after successful cleanup; poisoned states cannot be resumed.

## Explicit prepared generation

`PipelineReferenceGenerator(pipeline, text, ...)` supports ordinary single,
equal-length and opt-in ragged GLM text generation. For the independently pinned
native-tool grammar, use `PipelineToolReferenceGenerator` with `GlmToolText` and
its `generate_tools` / `generate_batch_tools` methods. These adapters reuse the
existing tokenizer, mapped vocabulary, parser, context limits, usage accounting,
cooperative cancellation and abandoned-iterator cleanup. Returned tool calls are
never executed. Close abandoned iterators and call runner cleanup before reuse.

Admission is explicit: ordinary complete-model generators still reject pipeline
objects, and these pipeline adapters reject individual stages. Ragged cohorts
execute sequentially under the shared cache limit. Failed aggregate releases
block a new request until cleanup succeeds. Generation tests compare real packed
pipeline text and usage with a whole-model floating control; tool parser tests
use controlled token selection after real stage execution, without claiming
trained tool-call competence. CLI/HTTP/network integration is still separate.

## Checkpoint construction and command

`load_reference_pipeline(path, layer_splits, ...)` audits the complete snapshot
before any tensor payload reads and applies one global weight budget: all retained
stage weights plus the largest source conversion buffer. It then loads each
assigned range under that same cap. Configuration, shard and index file identity
receipts are checked around admission and each construction step. These detect
ordinary concurrent changes; they are not content authentication against a
hostile writer. Header/Python storage, activation scratch and KV cache allowances
remain separate.

The existing local command accepts explicit full-indexer cuts:

```sh
python -m scripts.run_glm_reference /path/to/snapshot \
  --layer-splits 2 --prompt 'Hello' --max-new-tokens 16 \
  --context 1024 --weight-mib 256 --cache-mib 256
```

Replace the cut with valid internal full-indexer layer indices for the snapshot.
One to seven ordered cuts produce two to eight stages. Default command execution
is unchanged. Single text, tools, equal-length batches and opt-in ragged cohorts
use the explicit pipeline adapters when cuts are supplied. The initial output
identifies `execution: local_cpu_pipeline` and every stage range. This is one
local CPU process; it does not start workers, open a network or qualify hardware.

## Explicit loopback HTTP

The CPU reference server accepts the same GLM-only cuts:

```sh
python -m scripts.serve_cpu_reference glm /path/to/snapshot \
  --layer-splits 2 --context 1024 --weight-mib 256 --cache-mib 256 \
  --port 8088 --queue-size 4
```

The command preserves existing authentication and bounded serialized queue/SSE
behavior. Readiness, model status and response timing identify
`local_cpu_pipeline` and its layer ranges. Failed cleanup inside a stage makes
health unavailable just like a retained aggregate release. Cancellation and drain
use the complete pipeline owner; a successfully cleaned request leaves the owner
available for a fresh state. This remains local CPU execution, not multi-host or
GPU serving. Tests use real loopback subprocesses, authenticated JSON/SSE,
concurrent queued requests, drain and cooperative shutdown.

## Independent stage slots

`GlmStageSlots(stage, context, max_batch, max_cache_bytes=...)` reserves the
aggregate cache allowance for 1–4 independent slots before allocating any cache.
Each slot has its own cursor and can be prefetched in chunks, decoded in a
reordered subset, released and reused. `forward_layers` consumes finite CPU FP32
hidden chunks and exact contiguous position rows; `embedding` and `output_logits`
are separate first/final-stage boundaries. The underlying stage now exposes
`forward_activations` to execute only its owned layers, without embedding or
projecting a second time.

Bad input rejected before work preserves the owner. A partial row execution or
release failure fails the entire owner and attempts all slot releases. Failed
cleanup remains reachable for retry through `close`; a failed owner cannot
resume. Prepared weights remain caller-owned and immutable. This slot adapter
alone opens no sockets and does not widen existing host, GPU or HTTP admission.
Packed multi-stage tests compare unequal cursors, reordered decode and slot reuse
with independently expanded whole-model logits.

## Prepared authenticated host links

An explicit `HostTopology` with `quantization='fp8_reference'` can bind admitted
GLM configuration and full-indexer ranges. Wrap each independently loaded stage
in `GlmStageSlots`, authenticate/pin the caller-owned sockets, and construct
`PlannedHostStageChannel` plus `HostPipelineCoordinator` / `HostPipelineWorker`
owners as for the existing planned CPU topology. GLM uses its own configuration
and packed-precision identity; existing Qwen identities are unchanged.

The topology binds source/session identity, mapped/excluded vocabulary and all
slot/context/chunk/frame limits. Intermediate activations travel through the
coordinator; final workers return mapped sampled IDs. Release and STOP use owned
GLM slot cleanup. A failed link or stage discards the epoch. The existing raw
`HostCompletion` lifecycle also discards its session on abandoned generation;
normal completion releases slots for reuse. This differs from the reusable local
pipeline HTTP owner's request-level cancellation.

Two/three-stage local TLS tests compare full packed logits and greedy results
with independent floating controls, including unequal prefill, chunks, reordered
decode and slot reuse. Policy identity disagreement fails setup, and middle-stage
failure/peer loss closes links and caches. This prepared gate does not add a GLM
host builder, command or host HTTP adapter; existing Qwen-specific bootstrap and
HTTP admission reject GLM until those explicit integrations exist. Source bytes
remain an operator assertion unless separately verified; this is no physical
network, official-model or performance qualification.

## Snapshot-to-host construction

`build_glm_host_pipeline_worker(path, topology, index, transports, epoch=..., ...)`
constructs one process's assigned GLM range from a local snapshot. It validates
controls and atomically claims fresh ordered links using the same claim rule as
the Qwen builder. Per-process weight, projection, scratch, output and aggregate
slot-cache allowances apply. Configuration must match the topology exactly.

With `verify_source=True`, the bounded full-snapshot byte fingerprint must match
`topology.source_identity`, and its file receipt is checked after loading and
stage construction. Verification reads all selected snapshot bytes, including
text metadata; tensor decoding/loading still covers only the assigned range.
Without verification the source identity is an operator assertion. This local
verification is not remote attestation.

The builder creates no sockets, listeners or handshake threads. A
`GlmHostPipelineBuildError` owns claimed transports/channels and any stage until
`close()` succeeds; retry it when cleanup fails. Builder-owned checkpoint weights
are released after successful stage/cache close, while lightweight configuration
and range metadata remain available. Prepared caller-owned stage weights keep
the separate borrowing contract described above. Generated-snapshot tests cover
verified three-stage TLS execution, identity/configuration/budget mismatch,
file replacement and partial link cleanup failures.
