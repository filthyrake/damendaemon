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
