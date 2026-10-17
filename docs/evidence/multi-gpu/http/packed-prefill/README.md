# Resident distributed packed prefill

Historical qualification below predates the [adjacent Q4 stage implementation](../../../quantization/cuda/q4-arithmetic/README.md).
That implementation subsequently passes the expanded Q4 gate and enables Q4
packing by default for the qualified stage configuration.

Issue #51 tracks this work. The distributed service reuses the independently
qualified layer-stage packed math, retaining one owner thread for GPU/NCCL work.
FP16/Q8 enable packing at capacity greater than one. Q4 remains sequential by
default because of the [expanded numerical gap](../../packed-prefill/README.md).
`DAMENDAEMON_PARTITION_PACKED_PREFILL=0` disables packing; `1` explicitly enables it.
Both workers must agree on this setting.

Protocol version 4 carries each prompt's length, slot and generation in the
existing bounded control record. The flattened activation row count can differ
from the returned token count, which is one per prompt. Each group's total rows
are bounded by the configured context. Admission adds no coalescing delay.
Cancellation, acknowledged release and failure poisoning retain their existing
semantics. This is neither chunked prefill nor tensor parallelism.

## Physical qualification

Pinned Qwen2.5-0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`,
A2/A30 split 12/12, context 512, capacity four:

- FP16, Q8 and Q4 each pass 29 real HTTP checks against independent reference
  tokens. FP16/Q8 observe actual four-prompt packing; Q4 observes its sequential
  default. Coverage includes concurrent mixed completion lengths, slot refill,
  cancellation, streamed text/usage, rejected requests and 35 seconds idle.
- Injected output-stage packed-prefill failure processes 64 activation rows
  across four requests. All four clients fail; none gets final success.
- Injected four-row decode failure also fails every client. The streaming
  client receives an error without final usage, finish reason or `[DONE]`.
  Both failure probes leave zero active/queued work and health/new requests
  return 503. Each owned job exits cleanly from its expected error path.
- Both ranks reject all three startup disagreements: capacity, context and
  packing policy. These are six real rank-level rejection checks.
- 440 local tests pass, including prompt-budget splitting and the distinction
  between activation rows and failure-token count. These are correctness tests,
  not GPU performance evidence.

The failure probes require a usable CUDA context and transport. They do not
qualify process death, corrupted CUDA state, automatic recovery or replay.
Independent fresh healthy sessions are still required after a failed pair.

Reproduce healthy checks with `scripts/verify_distributed_service.py` and
`--batch-size 4`; use `verify_distributed_failure.py --operation prefill|decode
--stream` for fault probes and `verify_distributed_bounds.py` for disagreement
checks. All own bounded jobs. Run GPU jobs sequentially. The separate timing
harness supports `scripts/experiments/bench_distributed_service.py --compare
prefill`, fixing capacity four while alternating packing off/on/on/off.

## Matched A2/A30 HTTP measurement

Q8, context 512, capacity four, the same 16 raw prompt IDs and 32 output tokens;
sequential/packed/packed/sequential launches. Ten measured rounds per mode and
concurrency, with startup/capture and warmups excluded. Text and usage match.

| Concurrency | Packed | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | No | 17.62 | 120.84 | 263.85 |
| 1 | Yes | 17.35 | 119.01 | 267.83 |
| 2 | No | 30.14 | 180.76 | 350.23 |
| 2 | Yes | 26.97 | 176.17 | 357.69 |
| 4 | No | 47.32 | 228.83 | 554.16 |
| 4 | Yes | 49.61 | 211.97 | 593.44 |

At concurrency four, throughput rises 7.1% and request latency falls 7.4%, but
first text is 2.29 ms slower. Packing is not a universal latency improvement.
Single-request execution uses the same underlying path in both modes; its small
difference is variability. Existing services remain on the shared GPUs. This
compares packing policies within one precision and placement, not engines,
hardware, RDMA, or single-process versus distributed serving. Final allocations
return to A2 0 MiB and the existing A30 service at 13,403 MiB.

## A2/3090 network proof and matched measurement

The same Q8 revision and context pass 29 HTTP checks across NCCL sockets. The
initial four requests each report prefill group size four, and six acknowledged
heartbeats span the 35-second idle interval. A 64-row output-stage packed-prefill
failure makes all four clients fail and closes the service; both owned rank jobs
exit cleanly from the expected failure path. The subsequent benchmark launches
are fresh healthy sessions, not automatic recovery of the failed pair.

The fixed-capacity off/on/on/off measurement repeats the same 16-input/32-output
workload and ten measured rounds per mode/concurrency. All text and usage match.

| Concurrency | Packed | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---|---:|---:|---:|
| 1 | No | 18.28 | 130.76 | 243.74 |
| 1 | Yes | 19.34 | 133.00 | 240.09 |
| 2 | No | 28.08 | 188.88 | 332.07 |
| 2 | Yes | 36.74 | 188.94 | 335.09 |
| 4 | No | 51.34 | 248.20 | 507.24 |
| 4 | Yes | 46.07 | 220.71 | 568.63 |

Concurrency four gains 12.1% throughput, with 11.1% lower request latency and
10.3% lower first-text latency. Concurrency two has essentially unchanged
throughput/request time but first text is 8.66 ms slower. Single-request
differences again reflect variability on the same execution path. The packing
flag remains available for workload-specific latency/throughput choices.

The peer-only firewall rules had automatic 600-second expiry. They opened at
04:11:29 UTC on the lab and 04:11:33 on the desktop, and were removed early at
04:15:49/04:15:53 on 2026-10-06. Listings confirmed removal while preserving the
desktop's existing subnet rule. Final GPU allocations were A2 0 MiB, existing
A30 service 13,403 MiB, and existing 3090 service 10,835 MiB. No permanent or
router rules changed. This is socket transport evidence; RDMA remains untested.
