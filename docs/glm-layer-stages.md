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
