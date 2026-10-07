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
