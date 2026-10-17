# Bounded CPU Qwen3.5 checkpoint loading

`damendaemon.cpu_hybrid_checkpoint.load_cpu_checkpoint` connects local floating
safetensors snapshots to `build_cpu_pipeline`. It neither selects an accelerator
nor registers a serving model. This is a CPU reference path; fixture parity does
not qualify trained-model quality, practical context, latency or throughput.

```python
from damendaemon.cpu_hybrid_checkpoint import load_cpu_checkpoint
from damendaemon.cpu_pipeline_build import build_cpu_pipeline

model = load_cpu_checkpoint(snapshot, quantization='q4_0', max_context=512,
                           max_weight_bytes=256 * 1024**2,
                           max_workspace_bytes=64 * 1024**2)
pipeline = build_cpu_pipeline(model, stage_budgets)
try:
    result = pipeline.prefill(token_ids)
finally:
    pipeline.close()
```

The source must have a dense `qwen3_5_text` root configuration, default partial
RoPE, gated full attention and the existing supported linear-attention geometry.
Only F16, BF16 and F32 source tensors are accepted. The exact text inventory must
use one consistent `model.` or `model.language_model.` prefix; visual/MTP tensors,
quantized sources, missing/extra tensors and shard-index disagreement reject.
Multimodal wrapper configurations are not admitted by this text-only entry point.

The loader fuses gate/up and gated Q/K/V in source order, adds the Qwen offset to
RMS norm weights in FP32, and leaves gated linear-attention norms unchanged.
Untied heads require `lm_head.weight`. Tied snapshots must omit that redundant
source tensor; FP32 heads alias embeddings, whereas quantized heads own packed
values and FP16 scales. Q4/Q8 projection widths must be divisible by 32. Embeddings
and nonmatrix parameters stay FP32; matrix projections become canonical Q4_0 or
Q8_0. These conversions use the existing quantization conventions.

Before any tensor payload read, the loader admits the configuration, context,
whole source inventory and all tensor shapes/dtypes. `max_weight_bytes` bounds
retained weights plus scales; `max_workspace_bytes` bounds a conservative
32 bytes per largest fused tensor element for source/cast/fusion/quantization
overlap. These are separate allocations, so the loading envelope is their sum.
It processes one fused group at a time, never materializing the whole floating
checkpoint. Stage cache/rotary/projection budgets apply separately during pipeline
construction. The returned model owns source weights; closing a pipeline does
not destroy them. Release that model when its storage is no longer needed.

Metadata/Python/allocator overhead is outside tensor budgets: configuration is
capped at 1 MiB, index at 16 MiB, headers at 16 MiB each/64 MiB total and open
shards at 256. Payloads are copied through owned descriptors and file-stamp
changes reject; all descriptors close on success and failure. Snapshot files
must remain stable during loading. This is mutation detection, not cryptographic
checkpoint attestation.

Tests load actual tiny two-shard checkpoints in all three floating source dtypes
and output modes. Independent mappings, nonzero norm offsets, packed scales,
tied aliases, chunked prefill and reordered pipeline decode agree. Invalid
metadata/capacity fails before payload reads; injected read failure, nonfinite
values and file mutation exercise descriptor cleanup. GPU admission is unchanged.

## Stage-local source residency

`load_cpu_stage_checkpoint(snapshot, start, end, ...)` accepts the same precision,
context and loader budgets but retains only a contiguous half-open layer range.
The first range owns embeddings; the last owns final norm and output head; middle
ranges own neither. Use the returned partial model with
`CpuHybridStage(model, start, end, ...)`. Its `layer_range` records the requested
range. Whole-pipeline construction rejects partial inventories.

All source headers, shapes, dtypes and the shard index are still validated against
the complete checkpoint. Only selected payloads are copied, converted and scanned
for finiteness. Consequently a malformed excluded header rejects, while an
excluded nonfinite payload is deliberately not inspected; this loader does not
attest to excluded tensor values. Retained and conversion budgets cover the
selected range, allowing it to fit when the complete source model cannot.

For a tied final head, the loader reads the source embedding directly into the
head's selected representation. It does not retain an unused embedding tensor.
When one range owns both endpoints, FP32 aliasing and single-source read behavior
remain unchanged. Independent range loads have independent source ownership;
they are not automatically assigned a common in-process pipeline identity.
Authenticated worker setup still requires the caller's explicit model/session
identity and matching configuration. No deployment or automatic partitioning is
performed.

Tests track exact payload reads across first/middle/last ranges, compare selected
weights/scales and byte accounting to whole loads, and compare split execution
through chunked prefill/reordered decode. FP32, Q4 and Q8, tied/untied heads and
BF16 source rounding are covered; these are CPU fixture proofs.
