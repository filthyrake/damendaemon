# Continuous decode batching across a resident pair

Tracked in GitHub issue #51. `distributed-serve --batch-size 4` admits up to four
requests into independent KV slots. Each prompt is processed separately; active
requests then share decode passes. Released slots admit new prompts while other
requests continue. Both ranks acknowledge release before final success or slot
reuse. The default remains one slot. Both workers must specify the same capacity,
context, precision and snapshot; startup compares a fixed-size session digest.

## A2/A30 correctness

Pinned Qwen2.5-0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`,
12/12 split, context 512, native stage graphs. FP16, Q8 and Q4 each pass **25
real HTTP checks**. These cover independent reference continuations, repeated
prompts, shrinking concurrent batches, more concurrent requests than slots,
streaming usage, cancellation/reuse, unsupported-input recovery and repeated
chat. Multi-request decoding is observed in the returned timing records.
The output worker does not load the reference trace.

The local regression suite passes 428 tests. Targeted controls additionally
prove reordered/refilled slots, cancellation of one request during a shared
decode, failure propagation to every active request, shutdown release, one-token
completion without decode, and withheld final success until release is
acknowledged. CPU controls are not GPU performance evidence.

Reports `qwen2-distributed-http-batch4-{none,q8_0,q4_0}.json` contain source and
reference hashes. Reproduce with `scripts/verify_distributed_service.py` and
`--batch-size 4`; its default launcher owns a bounded torchrun process group.
These lifecycle runs use zero idle delay; the original single-slot evidence
separately qualifies 35-second idle residency.

## Matched Q8 HTTP measurement

`qwen2-distributed-http-batching-benchmark.json` compares capacity one versus
four on the **same A2/A30 pair**, prepared weights, precision, context 512,
16 input token IDs and 32 output tokens. Four separate launches run in capacity
order 1,4,4,1. Each concurrency gets a warmup plus five measured rounds per
launch, yielding ten rounds per mode. Startup and graph capture are excluded;
HTTP scheduling, text delivery and release acknowledgement are included.
Generated text and usage match exactly across every mode and concurrency.

| Concurrent requests | Capacity | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 17.77 | 120.67 | 264.31 |
| 1 | 4 | 17.88 | 120.76 | 264.00 |
| 2 | 1 | 74.45 | 176.99 | 270.06 |
| 2 | 4 | 26.65 | 174.01 | 366.06 |
| 4 | 1 | 196.80 | 298.89 | 268.69 |
| 4 | 4 | 43.93 | 226.17 | 562.08 |

At concurrency four, batching improves aggregate throughput by **2.09x** and
reduces median first text by **77.7%**, with essentially unchanged single-request
latency. This is an optimization comparison within damendaemon, not a comparison with
another engine, another GPU, or RDMA. It does not imply each request decodes
faster in isolation. Reproduce with
`scripts/experiments/bench_distributed_service.py`; use explicit GPU UUIDs and
run GPU jobs sequentially without preparation or compilation in the background.

The A2 returned to 0 MiB and the A30's existing service remained at 13,403 MiB.
No firewall changes were needed for these same-host runs. The narrow model,
context and greedy-only limits remain; this adds neither tensor parallelism nor
distributed model switching, sampling, prompt packing or chunked prefill.

## A2/3090 network batching

The batched Q8 service passes all 25 HTTP checks over NCCL Socket between the
A2 coordinator and desktop 3090. After 35 seconds idle, six acknowledged
heartbeats and an independent continuation confirm that the resident worker is
still usable. Both workers exit zero. The prepared Q8 identity remains
`130f2c0e86933c57790f0efbe9d1e5249c0c3df768a83d4081bedf32a3e9b046`.

The network benchmark uses the same pinned model, 16 raw input IDs, 32 output
tokens, context 512 and capacity order 1,4,4,1 as the same-host comparison.
Every measured response has identical text and usage across capacities. These
runs are sequential; no other test or remote preparation runs during the timed
samples. Existing GPU services remain present on the shared hosts.

| Concurrent requests | Capacity | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 17.86 | 128.19 | 248.87 |
| 1 | 4 | 17.89 | 126.41 | 252.34 |
| 2 | 1 | 78.50 | 190.03 | 253.69 |
| 2 | 4 | 25.76 | 179.08 | 355.45 |
| 4 | 1 | 213.34 | 325.08 | 252.08 |
| 4 | 4 | 44.57 | 232.39 | 546.39 |

At concurrency four this is **2.17x aggregate throughput**, **79.1% lower
median first text**, and **28.5% lower median request latency**. A single
request remains around 127 ms. The small single-request difference is not
claimed as a speedup. These are same-hardware capacity comparisons; neither
this result nor the same-host table proves which GPU placement is best. This
network has no qualified RDMA device; the transport is ordinary NCCL sockets.

`qwen2-network-http-batch4-q8.json` records correctness/idle qualification;
`qwen2-network-http-batching-benchmark.json` records every warmup, measured
trial, model identity and source hash. Use the benchmark's `--worker` mode
under bounded two-host torchrun launches, changing only `--batch-size` between
runs. The `summarize` helper computes the same metrics for either topology.

Native Q4 also passes all 25 network HTTP checks, with zero idle delay; both
workers exit zero (`qwen2-network-http-batch4-q4.json`). Temporary peer-only
TCP rules had 600-second automatic expiry. Lab rules were active from 03:06:12
to 03:11:42 UTC; the desktop rule from 03:06:16 to 03:11:46 on 2026-10-06. All
were removed early. The original desktop rule remains, with no permanent or
router changes. Final allocations are A2 0 MiB, A30 existing service 13,403 MiB
and desktop existing service 10,835 MiB. See `network-window.json`.

## Startup mismatch rejection

`qwen2-distributed-bounds.json` records two real A2/A30 jobs: capacity 4 versus
1, and context 512 versus 256. Both ranks reject each mismatch with the explicit
session-bounds error before constructing the command channel, then exit zero
through the expected failure path. GPU allocations return to their baseline.
Reproduce with `scripts/verify_distributed_bounds.py`; each owned torchrun job
has a 180-second outer deadline. This proves rejection/cleanup, not performance.

The subsequent [output-stage exception proof](../failure/README.md) records
protocol version 3, a real failure uncovered by GPU testing, explicit failure
reporting, fresh precision gates and repeated healthy-path measurements.
