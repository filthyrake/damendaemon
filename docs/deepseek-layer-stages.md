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
block/model/generator/HTTP cleanup retry are covered. Complete pipeline integration is described below; host transport remains a subsequent gate.

## Complete local pipelines

`deepseek_checkpoint_pipeline.load_reference_pipeline(snapshot, [cut, ...])`
constructs 2..8 complete contiguous ranges after global retained-plus-largest-source
weight admission. File receipts surround each range and final construction; they
detect ordinary concurrent changes, not authenticate model content.
`DeepSeekLayerPipeline` carries original IDs through every range, owns one aggregate
cache budget, and exposes prefill, one-token decode and shared row reorder. Partial
execution poisons every range; failed release stays owned for retry.

`DeepSeekPipelineReferenceGenerator` explicitly admits this complete CPU FP32
pipeline and preserves the pinned native text/DSML tool paths, strict schemas,
equal/ragged cohorts and token-boundary cancellation. Ordinary whole-model and
partial-stage admission remain separate.

Use `python -m scripts.run_deepseek_reference SNAPSHOT --prompt TEXT
--layer-splits 1 2` or `python -m scripts.serve_cpu_reference deepseek SNAPSHOT
--layer-splits 1 2` for explicit local pipeline execution. Cuts must match the actual
layer count. The generation command keeps its projection `--workspace-mib`; HTTP
uses `--projection-workspace-mib`. Pipeline `--output-mib` separately bounds residual
streams and logits. Existing context/cache/weight bounds, HTTP authentication,
optional serialized queue, SSE, drain and signal cleanup apply. Metadata reports
`local_cpu_pipeline` and exact stage ranges. Whole-model defaults are unchanged.

Packed arithmetic comparisons, source receipts, native single/equal/ragged/tool
requests, real subprocess commands and HTTP JSON/SSE/queue/drain controls are tested.
CPU fixtures do not qualify official weights, practical context, GPUs or speed.

## Independent request slots

`DeepSeekStageSlots(stage, context, max_batch=4, max_cache_bytes=...)` reserves
capacity for every admitted slot before allocating compressed-attention state.
It shares checked cursor, release and failure ownership with GLM/Flash slots,
but explicitly requires `forward_layers(..., token_ids=original_ids)`. Both IDs
and full residual streams are checked before mutation. New slots use stage
prefill; continuation chunks use cached per-token decode and preserve every
intermediate activation for the next stage. This bounded reference path makes no
chunked-prefill speed claim.

Slots may have unequal cursors and be selected in any distinct order. Release
clears only selected requests and permits reuse at position zero. Partial compute
or release failure retires the owner; retain it and retry close until all stage
and block cleanup succeeds. The final stage applies the learned head only through
`output_logits`. No token-bearing network transport is admitted by this interface.

## Prepared token-bearing host stages

`HostTopology` explicitly admits DeepSeek with `quantization='fp4_fp8_reference'`.
Its immutable identity includes all model options, complete contiguous ranges,
source identity, token exclusions, slot/context/frame limits and the DeepSeek
integer-token-before-activation protocol. Width includes every residual stream.
Other model families retain their existing wire format.

Each execution sends checked CPU int64 original IDs, then FP32 flattened residual
streams over already authenticated `PlannedHostStageChannel` links. Receivers check
frame shape/type and mapped/excluded IDs before layers run. Intermediate stages
return complete streams; only the final stage applies the learned head and samples.
All participants acknowledge completion. Invalid frames, peer loss or partial
execution discard the epoch and retain failed cleanup rather than reuse state.

`build_deepseek_host_pipeline_worker` claims fresh prepared transports, loads only
the assigned stage and owns both checkpoint weights and caches. It exposes separate
weight, projection workspace, output and cache limits. Optional `verify_source=True`
checks a bounded complete snapshot fingerprint and rechecks file receipts after
loading/construction. Claimed construction failures retain retryable cleanup through
`DeepSeekHostPipelineBuildError`. No network listeners or native host command/HTTP
admission are supplied by this builder. Tests use real pinned loopback TLS and tiny
packed CPU snapshots; remote deployment and performance remain unqualified.

## Native host generation

`deepseek_host_generation.native_session_identity(text, verified=...)` binds a
detached validated ByteLevel tokenizer, EOS, the pinned V4 text/DSML renderer and
source-verification mode. Supply that identity to every topology participant.
`HostDeepSeekGenerator` accepts only this matching, ready, unclaimed DeepSeek
coordinator. It supports native `generate`/`generate_batch`, ordinary/thinking text,
explicit ragged batches and optional declared tools with existing strict schemas.
Tool declarations are snapshotted before dispatch; results return calls without
executing them.

Healthy requests release all slots before final success and allow reuse. Parser
errors, partial computation, cancellation and abandoned streams discard the host
epoch. Parser/transport cleanup remains owned for retry. The shared native lifecycle
checks cancellation after parser output and retains each parser as it is constructed,
so closing during buffered-call finalization cannot publish a successful tool call,
and failure constructing a later parser closes those already created.

This is explicit prepared-host admission, separate from generic local reference
HTTP. Native host command/HTTP integration follows. CPU controls, real local TLS
and controlled output tokens after packed arithmetic test format and lifecycle;
they do not establish trained-model tool quality or accelerator performance.
