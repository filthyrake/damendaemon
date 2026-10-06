# Output-stage exception reporting

Tracked in GitHub issue #51. A real fault injection against revision `76d6b2b`
exposed an unhandled boundary: an output-stage exception during a four-row
decode left the coordinator waiting for tokens. NCCL's watchdog timed out after
30,033 ms and aborted rank zero with SIGABRT. The owned job ended nonzero and
GPU allocations returned to baseline; local fake-pair tests had not exposed
this behavior.

Protocol version 3 reserves token value -1 for an output-stage execution
exception. The output worker reserves a small failure tensor during startup
and sends it over the existing fixed-size token reply when an exception occurs
after activation receipt. The coordinator rejects that payload, poisons the
session, and fails active/queued requests. Both workers require matching
versions; versions 1 and 2 are rejected. Successful execution gains no extra
collective or acknowledgement round trip.

This only handles exceptions while CUDA and the transport are still usable.
A corrupted CUDA context, broken transport, worker death, or a coordinator-stage
failure can still require the process owner's deadline/termination. There is
no automatic restart, transparent replay, or claim that HTTP survives every
fatal GPU failure.

## Real A2/A30 qualification

The final implementation passes two owned GPU jobs, injecting an exception in
the output stage during a 16-token prefill and during a four-request decode.
Each job uses Q8 Qwen2.5-0.5B, context 512, split 12/12, capacity four.

- All four concurrent non-streaming requests fail with HTTP 500 or 503; no
  completion text/usage is returned as a successful response.
- Status reports failed/closed, zero completed requests, no active or queued
  work. Health and a new HTTP completion both return 503.
- The expected failure paths exit zero in the qualification harness; both GPU
  allocations return to baseline. Production still raises the execution error.
- Fresh subsequent FP16, Q8 and Q4 sessions each pass all 25 independent HTTP
  correctness/lifecycle checks. This is explicit fresh-process recovery, not
  automatic worker restart.

The 431-test local suite passes, including reserved/invalid token rejection,
no extra transfer for successful output selection, and use of the already
allocated failure payload before propagating the original error. CPU controls
are not GPU performance evidence.

Reproduce with `scripts/verify_distributed_failure.py --operation prefill` or
`--operation decode`. The parent owns a torchrun process group with a 180-second
outer deadline. `--worker` supports an externally bounded cross-host launch;
`--stream` additionally checks an in-progress SSE response for an error without
a finish reason, usage or `[DONE]` success marker. The reports include source
hashes and the actual injected operation/shape.

## Healthy-path measurement with protocol version 3

The same counterbalanced capacity 1,4,4,1 Q8 HTTP benchmark was repeated with
the final failure-buffer implementation: identical A2/A30 pair, prepared
weights, context 512, 16 input IDs, 32 output tokens, ten measured rounds per
mode/concurrency. All generated text and usage match. Startup/capture is
excluded; HTTP scheduling and acknowledged release are included.

| Concurrent requests | Capacity | Median first text, ms | Median request, ms | Aggregate output tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 18.83 | 122.87 | 259.48 |
| 1 | 4 | 17.63 | 121.01 | 263.38 |
| 2 | 1 | 76.63 | 178.35 | 267.11 |
| 2 | 4 | 25.51 | 176.20 | 361.62 |
| 4 | 1 | 199.98 | 302.06 | 267.04 |
| 4 | 4 | 43.32 | 227.92 | 556.99 |

Batching retains a **2.09x** aggregate throughput gain at concurrency four.
This is a matched capacity comparison within version 3, not a counterbalanced
version-2/version-3 speed comparison. Existing GPU services remain untouched;
A2 ends at 0 MiB and the A30's original service at 13,403 MiB.

## Streamed failure over the A2/3090 network

The same exception signal passes a real NCCL Socket test between the A2 and
3090. The output stage raises during a four-request decode. Three buffered
requests return HTTP 500. The fourth already has an HTTP 200 event stream; it
receives its initial text followed by an explicit error frame, with no finish
reason, final usage, or `[DONE]` marker. Health and a subsequent HTTP completion
return 503. Status is failed/closed with zero completed, active or queued work.
Both qualification workers exit zero through the expected exception path.

The paired `qwen2-network-http-failure-v3-{coordinator,worker}.json` reports
record the response frames, source hashes and actual four-row injected decode.
This establishes graceful reporting for an output-stage exception while CUDA
and NCCL remain usable. It does not establish survival of arbitrary process,
GPU or network failures.

Fresh Q8 and Q4 network sessions subsequently pass all 25 independent HTTP
checks each (`qwen2-network-http-v3-{q8,q4}.json`). This is explicit fresh-worker
recovery. All six workers across the three jobs exit zero. The peer-only rules
had 600-second automatic expiry and were removed early: lab 03:25:48–03:29:17
UTC, desktop 03:25:44–03:29:24 on 2026-10-06. Original rules are preserved;
A2 returns to 0 MiB, A30 remains at 13,403 MiB and desktop at 10,835 MiB.
No permanent or router changes were made; see `network-window.json`.
