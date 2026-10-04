# Bounded paged KV cache

Issue #27. The independent CUDA implementation uses sixteen-token pages, a
fixed-capacity physical pool, and a mutable logical-to-physical page table.
Page ownership is shared across layers but isolated by request. Native rotary
writes and decode attention resolve page addresses directly; CUDA graphs retain
stable buffer pointers while the table changes.

Host admission reserves prompt plus maximum output capacity before prefill.
Requests temporarily unable to fit wait in FIFO order while existing requests
continue. Oversized requests fail before inference. Pages are returned on all
completion, cancellation, prefill-error and group teardown paths. Direct
single-request generation holds the same reservation through generator closure.

Initial A2 evidence:

- Twelve independent fragmented-addressing cases cover FP16/FP32, dimensions
  64/96/128, GQA, page boundaries, windowed/full attention and NaN-filled unused
  storage. Rotary writes and attention match tensor oracles.
- All 115 existing kernel checks pass after the optional addressing extension.
- Qwen2.5-0.5B Q4 passes 82 reference checks with both 8192-token and reduced
  3072-token physical pools at 2048 context and four logical slots.
- Accurate Qwen3-0.6B Q4 passes its independent reference with a 4096-token pool.
- Real HTTP testing uses a 512-token pool with four 1024-context slots. Four
  requests that can only fit one at a time complete identically; peak ownership
  is 336 tokens. Oversized requests are rejected. A directly closed stream is
  cancelled after one output token. Ownership returns to zero, and model
  switching plus host restore preserves output and frees pages.

This is opt-in CUDA dense support. Prefix snapshots and interleaved prefill are
currently rejected in combination; no KV quantization or shared-prefix page
references are implemented. Actual memory and matched throughput measurements
are recorded alongside their JSON evidence when complete.

## Initial memory and throughput measurements

A2, Qwen3-0.6B Q4 with accurate FP32 arithmetic/KV, four slots, context 2048,
1024-token prompt and eight decode steps: fixed-slot KV storage is
1,879,048,192 bytes, versus 469,762,048 with a 2048-token page pool (75% less).
Resident PyTorch allocations fall from 2,538,435,584 to 1,129,151,488 bytes.
Generated token IDs are identical. These are allocator figures, not total driver
memory. Physical page capacity is allocated up front; ownership is dynamic.

Matched Qwen2.5-0.5B Q4 serving, same checkpoint/template, 2048 context,
64 requested output tokens and five warm measured rounds:

| Concurrent requests | Fixed tokens/s | Paged tokens/s | Fixed TTFT ms | Paged TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 205.34 | 203.15 | 28.20 | 31.47 |
| 2 | 353.31 | 353.37 | 49.10 | 36.64 |
| 4 | 644.25 | 599.80 | 71.70 | 88.50 |

The initial implementation loses about 7% throughput at four requests. It stays
opt-in; the memory reduction is not a claim of a universal speed improvement.
Prefill gathers and page-table updates are being profiled as follow-up work in
issue #27. The unit/lifecycle suite has 148 passing tests.

## Fused prefill materialization

The initial profile identified 384 extra page-gather/copy launches per packed
four-request prefill (24 layers). The rotary/cache-write kernel now optionally
also emits current-token K/V, so full prefill consumes views directly. Decode
continues to read physical pages. This removes the gathers without changing
cache precision or ownership. The fragmented kernel oracle additionally checks
materialized K/V against the cache and independent rotation.

Qwen2/Qwen3 Q4 independently pass 82/84 reference checks after this change.
Matched serving results (`qwen2-paged-direct-kv`) are 194.89 / 346.20 / 605.62
tokens/s at concurrency 1/2/4, with median TTFT 30.07 / 49.76 / 72.89 ms.
The four-request TTFT improves from the initial 88.50 ms, but throughput remains
below the earlier fixed-cache result; this is not evidence of a throughput win.
Profiles are diagnostic, separate from the unprofiled HTTP measurements.
