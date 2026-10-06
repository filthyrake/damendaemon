# Resident distributed HTTP qualification

Tracked in GitHub issue #51. `distributed-serve` exposes the existing loopback
chat/completion API over a resident two-rank pair. It is a greedy sequential
queue with a single model and slot; it is not continuous batching, distributed
hot swapping or general architecture support.

HTTP threads validate and queue requests. Only the owning thread touches the
GPU pair. Cancellation releases the current slot after its in-flight operation;
final success waits for both ranks to acknowledge release. Queue limits and
admission deadlines are explicit. An admitted prefill is no longer subject to
the queue deadline. Execution/transport failure poisons the pair and fails
queued work; automatic worker restart/replay is not implemented.

Control protocol version 2 adds canonical acknowledged PING records. They
advance sequence numbers without changing slot lengths or generations. The
coordinator sends one every five idle seconds, preventing the output worker's
30-second collective wait from expiring. Both workers need matching versions.

## A2/A30 real GPU proof

Pinned Qwen2.5-0.5B, 12/12 layer split, context 512, one active request. Existing
A30 service allocations were preserved at 13,403 MiB; the A2 returned to 0 MiB
after every completed job. No host firewall change was needed for these tests.

| Precision | HTTP checks | Idle check |
|---|---:|---|
| FP16 | 19 passed | Immediate follow-up |
| Q8 | 19 passed | 35 seconds, six acknowledged heartbeats |
| Q4 | 19 passed | Immediate follow-up |

Each gate uses independent reference continuations, varied/reused prompts,
concurrent queued requests with different output lengths, streaming usage,
client cancellation followed by another prompt, malformed/unsupported inputs,
chat repetition and a final independent request. The output worker reads no
reference trace. Failed unsupported-feature requests do not poison a healthy
pair. Sampling changes, stops, logprobs and model management are rejected.
The Q8 prepared identity remains
`130f2c0e86933c57790f0efbe9d1e5249c0c3df768a83d4081bedf32a3e9b046`.

A separate existing `distributed-run --requests-file` job passes four exact
independent Q8 continuations after the protocol update
(`qwen2-distributed-v2-cli.json`). The 419-test local suite includes acknowledged
release before final publication, cancellation/reuse, single-thread GPU ownership,
queue limits/expiry, admission versus first-token deadlines, idle heartbeats,
stale protocol rejection and failure propagation. CPU control tests are not GPU
performance evidence.

Reproduce with `scripts/verify_distributed_service.py`. Its default launcher
owns a bounded same-host torchrun process group using `--run-path`. For a network
run, invoke its `--worker` mode under paired torchrun commands with an outer
process deadline. Reports contain model/reference identity and source hashes.
These are correctness/lifecycle gates, not matched performance measurements.
Cross-host HTTP qualification is recorded below, separately from the earlier
distributed CLI network proof.


## A2/3090 network HTTP proof

The same Q8 service passes all 19 checks over NCCL sockets between labnode3's
A2 and the desktop 3090. The coordinator alone reads the independent trace.
After 35 idle seconds and six acknowledged heartbeats, an independent request
still succeeds. Streaming, cancellation followed by slot reuse, concurrent
queued requests, bad-input recovery and repeated chat all pass. Both workers
exit zero. This establishes the bounded HTTP path, not batching, hot swapping,
RDMA or a network speedup.

Only peer-specific temporary TCP rules were added, each with 600-second automatic
expiry. Lab rules were confirmed at 02:46:51 UTC and removed at 02:49:24; desktop
rules at 02:46:40 and 02:49:27 on 2026-10-06. Original rules were preserved, with
no permanent/router change. Final GPU allocations match their starting values:
A2 0 MiB, A30 existing service 13,403 MiB, desktop existing service 10,835 MiB.
See `qwen2-network-http-q8.json` and the window/cleanup sidecar.
