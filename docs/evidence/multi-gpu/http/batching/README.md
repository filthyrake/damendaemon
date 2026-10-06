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
latency. This is an optimization comparison within dllm, not a comparison with
another engine, another GPU, or RDMA. It does not imply each request decodes
faster in isolation. Reproduce with
`scripts/experiments/bench_distributed_service.py`; use explicit GPU UUIDs and
run GPU jobs sequentially without preparation or compilation in the background.

The A2 returned to 0 MiB and the A30's existing service remained at 13,403 MiB.
No firewall changes were needed for these same-host runs. The narrow model,
context and greedy-only limits remain; this adds neither tensor parallelism nor
distributed model switching, sampling, prompt packing or chunked prefill.
