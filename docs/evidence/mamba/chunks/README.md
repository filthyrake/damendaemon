# Mamba chunked prompts and recurrent prefix snapshots

Native CUDA and Metal continuation convolution preserves prior history across
multi-token chunks. Each program/thread owns a channel and writes history only
after all outputs are computed. Existing native selective scans continue SSM
state across chunks; nonfinal chunks omit output normalization/projection.
Scheduler decode can interleave between chunks. Both options remain opt-in.

`--prefill-chunk-size 128` and `--prefix-cache-tokens 4096` now work for Mamba.
Snapshots copy all convolution and SSM state into independent storage, plus the
last logits. A snapshot can resume only its **entire stored prefix**: divergent
or shorter prefixes cannot rewind recurrent state. Prompts larger than the cache
token budget are skipped, not truncated. The token/entry LRU bounds and available
GPU memory check apply; snapshot state is fixed-size, 2,949,120 bytes for 130M
and 7,864,320 bytes for 370M, plus logits. Model eviction discards its snapshots.

Independent proof:
- 24 CPU kernel-oracle cases per GPU, including continuation lengths 1/2/7/33,
  convolution widths 1/4/8, odd channels, packed projections and reordered slots.
- Full CPU-reference recomputation at 12 boundaries per model/device trace:
  130M FP32 on CUDA/Metal, 370M Q8 on CUDA. Logits and selected layers' full state
  agree through 2048 tokens, including interleaved chunks, decode after restore,
  suffix continuation, snapshot isolation, state reset and invalid continuation.
- Real HTTP on both GPUs: streaming, concurrency, cancellation, model swapping
  and host restoration with both options enabled; CUDA service uses Q8.
- Separate cached/uncached HTTP comparisons agree on exact and extended prompts,
  four concurrent hits and model alias/eviction isolation.

Matched 130M FP32 A2 stream-interference experiment, 1536-token incoming prompt,
64-token active stream, three warm trials, identical greedy output:

| Chunk size | Median worst stream gap ms | Incoming request ms | Active request ms |
|---|---:|---:|---:|
| full | 162.41 | 205.65 | 413.00 |
| 128 | 23.00 | 277.92 | 452.21 |
| 256 | 35.71 | 235.10 | 426.91 |

Chunking reduces stalls but adds total work/latency; it is not a throughput win.
Warm 1536-token prefix, chunk128, five HTTP trials, first-token milliseconds:

| GPU | Uncached | Exact hit | Extended prefix hit |
|---|---:|---:|---:|
| A2 | 188.47 | 1.47 | 15.34 |
| M4 | 415.22 | 1.21 | 9.09 |

These compare damendaemon configurations, not competitors. Apple timings are a single
laptop run and may vary. Raw measurements, configuration and samples accompany
this file. Unit suite: 177 passing tests, including existing attention-prefix
rewind behavior and new recurrent no-rewind/memory-budget cases.
