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

This is opt-in CUDA dense support. No KV quantization or shared-prefix page
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

Llama Q4 passes 82 reference checks at context 2048 and Mistral Q4 passes 97
at context 4352, including its sliding-window boundary. The fused prefill
profile measures 15.16 ms GPU time per four-request prefill, versus 15.98 ms
for the initial paged path and 15.11 ms for fixed slots. Decode is 4.37 ms.
These are instrumented GPU times, not end-to-end latency.

## Interleaved prompt chunks

Paged request admission now keeps a reservation across partial prompt chunks,
executes at most one chunk between decode steps, and releases ownership on
partial cancellation or failure. Prefix snapshots were initially unsupported;
the later integration is described below.
The 155-test local suite includes decode interleaving, capacity deferral,
partial cancellation, failed-chunk recovery and model-management barriers.

The reference replay retires all slots replaced by a prefill event before
replaying its chunks sequentially; otherwise an old long prompt temporarily
consumes pages that the event has already logically discarded.

With 128-token chunks, Qwen2/Qwen3/Mistral Q4 pass 82/84/97 independent checks.
The real 512-token-pool service proof also passes with 128-token chunks.
Llama Q4 matches every greedy token but fails one of 82 numerical checks:
relative RMS 0.0109906 exceeds the 0.01 limit (max absolute 0.08594).
The fixed-cache chunked path fails the identical check with identical error;
Llama Q4 chunking is therefore **not numerically qualified**, while full paged
prefill remains qualified. Both failed trace reports are retained.

Qwen2 Q4 streaming benchmark, A2, 2048-token pool/context, 1536-token prompt
joining a 64-output-token stream, three measured rounds and identical greedy
outputs in all modes:

| Prompt chunk | Median worst active-stream gap ms | Long request s | Short request s |
|---|---:|---:|---:|
| full | 124.08 | 0.167 | 0.446 |
| 128 | 36.25 | 0.437 | 0.673 |
| 256 | 52.01 | 0.295 | 0.567 |

Chunking reduces the worst stream interruption but increases total request
latency in this test. It remains opt-in. Offset chunks currently materialize the
live prefix; bulk page copy and prefix integration are the next optimization.

## Native bulk copies and prefix snapshots

A native CUDA kernel gathers or restores all K/V heads for a prompt prefix in
one launch per layer. Exact FP16/FP32 oracle checks cover fragmented pages,
lengths 1/15/16/17/127/513 and untouched storage outside the live prefix.
Offset chunks use this kernel in place of separate index-select and layout
copies. Qwen2/Qwen3/Mistral Q4 replay remains 82/84/97 passing checks.

The scheduler can now combine page reservations, chunking and bounded prefix
snapshots. Exact hits use saved logits; partial hits restore independent pages
and compute the remaining suffix. Snapshots occupy additional GPU memory under
the separate prefix token budget; this is copied reuse, not shared page
references. Restored requests retain their full admission reservation.
The 157 local tests include exact-hit cleanup and failed-restore recovery.

Qwen2 Q4 prefix reuse passes 48 independent FP32 eager-reference checks,
including exact/partial matches, divergent suffixes, slot reuse, decode after
restore and bounded eviction. The live HTTP comparison uses Qwen2 Q4,
2048-context slots, a 2048-token physical pool, 128-token prompt chunks,
1536-token prompts, eight output tokens and five warm trials per cache mode:

| Prefix snapshot budget | Repeated TTFT ms | Partial-prefix TTFT ms | Repeated HTTP total ms |
|---|---:|---:|---:|
| disabled | 330.13 | 330.11 | 371.09 |
| 4096 tokens | 3.30 | 30.53 | 44.20 |

Outputs match across cache modes, concurrent requests and model eviction/host
restore. These are warmed prompt-reuse results, not improvements to uncached
model throughput. The physical pool only fits one of these long reservations
at a time, so concurrent hits also exercise capacity deferral.
