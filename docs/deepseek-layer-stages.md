# Packed DeepSeek CPU layer stages

`deepseek_reference_loader.load_reference_stage(snapshot, start, end, ...)`
audits every main-model header before reading only the assigned contiguous
range. The first range owns embeddings; the final range owns the complete
hyperconnection head, normalization and vocabulary projection. FP4 routed experts
and FP8 projections/shared experts remain packed. MTP remains excluded. Selected
retained parameters plus the largest source conversion tensor must fit one weight
allowance before payload reads; projection workspace is admitted separately.

`DeepSeekLayerStage` is separate from ordinary `FloatingModel` admission. Its
CPU FP32 boundary carries both original CPU integer token IDs and finite
`[batch, tokens, streams, width]` activations. Token IDs remain necessary after
embedding because hash-routed experts depend on them. `prefill(tokens,
hidden=..., max_tokens=...)` embeds only at the first stage; later ranges receive
the preceding streams. `decode(tokens, state, position=..., hidden=...)` advances
one token per row. Only the last stage applies the complete learned head reduction.
Layers-only activation methods preserve streams even at the final range.

Stage prefill checks the aggregate storage for every owned sliding-window,
compressed vector, compressor pool and index cache before allocation. Output
storage, context, token bounds and stream geometry have separate limits. States
own a shared cursor and bounded row order; partial decode or reorder poisons the
range. Retain the state and close it until every cache releases successfully.
These are explicit CPU controls, not accelerator/context/performance qualification.

## Failed-prefill ownership

A block's post-attention FFN can fail after attention has returned its cache.
Previously, if cache close also raised, the cleanup error replaced the original
failure and no retry owner retained that attention state. Blocks now retain failed
attention cleanup, preserve the primary error, and retry cleanup before another
prefill. Models, generators and reference HTTP status include pending block
cleanup as well as previously completed layers. Partial stage construction keeps
both prior-layer state and the failing block's attention until release succeeds.

Tests compare separately loaded full stacks with two/three packed ranges over
window and 4/128-token compression boundaries, including reordered slots and a
hash-routed layer after a cut. Exact selected weight/cache limits, unowned-header
audits, invalid IDs/streams/cursors, layers-only boundaries, partial failures and
block/model/generator/HTTP cleanup retry are covered. No complete pipeline,
native command/HTTP pipeline admission or host transport is supplied by this gate.
