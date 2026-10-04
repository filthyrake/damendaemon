# Bounded CUDA prompt graph reuse

Issue #34. `DLLM_PREFILL_GRAPHS=1..8` enables a per-executor LRU of captured
single-prompt CUDA forwards. Default is0. `DLLM_PREFILL_GRAPH_MAX_TOKENS` defaults
to128 and accepts1..512. The cache key is exact prompt length and cache slot;
input IDs are copied into stable graph-owned storage before each replay.

Qualified scope: unquantized Qwen2/Qwen3 on CUDA with graphs enabled. Apple,
quantized models, other families, paged KV, packed/chunked prefills and direct
candidate scoring use their existing paths. These exclusions prevent accidental
capture of incompatible metadata or a temporary output head. Multi-request
serving can still benefit when admission performs a qualifying single prefill.

Each cache entry owns its input/position buffers, graph and logits. LRU eviction
happens before a new capture. No retained callback holds the executor. Captures
include three warmup forwards and synchronization and can delay active requests;
this is why the option remains off by default. Varying shapes may never amortize
capture. A count bound and maximum prompt length bound retained graph workspaces;
this is not a user-configurable byte budget or a latency guarantee.

## Correctness and resource lifecycle

A2 / torch2.6+cu124, pinned Qwen2.5-0.5B-Instruct
`7ae557604adf67be50417f59c2c2f167def9a775` (standard precision) and Qwen3-0.6B
`c1899de289a04d12100db370d81485cdf75e47ca` (`DLLM_QWEN3_PRECISION=float32`).
Both pass changed IDs, cache hit/miss, multiple slots, LRU eviction,128-token
boundary and129-token fallback, subsequent captured decode, and scoring bypass.
All tested prefill and subsequent-decode logits match eager exactly. Weak-reference
checks prove evicted graphs and unloaded executor/graphs are no longer retained.

After clearing the two-entry test cache and emptying unused allocator blocks,
reserved memory decreased by30MiB for Qwen2 and56MiB for Qwen3. Live allocation
fell by about0.58/1.16MiB respectively; allocator reservation and live tensors are
separate measurements. Both are recorded in the reports. These deltas depend on
shapes and allocator history and are not per-entry memory limits.

Complete prefill timings include token tensor creation/copy,15 synchronized
trials after3 warmups, same resident executor and input shape:

| Model | Tokens | Eager ms | Cached graph ms |
|---|---:|---:|---:|
| Qwen2 | 16 | 21.112 | 7.710 |
| Qwen2 | 128 | 22.976 | 11.354 |
| Qwen3 FP32 | 16 | 42.570 | 9.622 |
| Qwen3 FP32 | 128 | 45.935 | 31.032 |

First Qwen2 captures in the full-path test took roughly188–246ms, including
warmups, versus7.7ms warm replay at16 tokens. Repeated shapes need roughly a dozen
or more requests to amortize that first-capture cost; actual break-even depends
on shape and load. The separate512-token probe found only a small gain, so the
default capture ceiling is128. The probe's allocated-byte figure excludes the
reserved private-pool cost; use the full lifecycle reports for memory evidence.

## Matched HTTP comparison

Same A2, pinned Qwen2 checkpoint, standard precision,512 context,16 exact raw
input IDs,32 output tokens, batch capacity4, five measured rounds after warmup.
Graph cache disabled versus4 entries; no prefix caching. Both use this same
implementation, not a comparison against another engine. Command/environment
records and every trial are retained. All outputs agree across both settings
and all concurrency levels.

| Concurrent requests | Off tokens/s | On tokens/s | Off TTFT ms | On TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 125.72 | 138.66 | 34.21 | 11.24 |
| 2 | 213.99 | 234.03 | 34.07 | 20.43 |
| 4 | 401.83 | 425.01 | 60.42 | 44.63 |

Medians; throughput includes prompt work. These warm repeated-shape gains do
not establish general throughput parity or cold/hot-swap latency improvement.

`decoder-prefill-graphs-service.json` is a successful real HTTP lifecycle run
with four graph entries and Qwen3 FP32: one and two resident GPU models, repeated
hot swaps, stream/nonstream equality, invalid/load-failure recovery, concurrent
requests and disconnect recovery. Its timings include cold activation and are
not substitutes for the matched warm comparison above.
