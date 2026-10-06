# Resident distributed packed prefill

Issue #51 tracks this work. The distributed service reuses the independently
qualified layer-stage packed math, retaining one owner thread for GPU/NCCL work.
FP16/Q8 enable packing at capacity greater than one. Q4 remains sequential by
default because of the [expanded numerical gap](../../packed-prefill/README.md).
`DLLM_PARTITION_PACKED_PREFILL=0` disables packing; `1` explicitly enables it.
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
