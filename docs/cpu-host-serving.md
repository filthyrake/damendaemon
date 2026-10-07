# Authenticated CPU host HTTP runtime

`HostServingPair` and `HostDistributedRuntime` in `dllm.host_serving` connect a
prepared authenticated CPU coordinator to the existing bounded distributed
request owner. They reuse the ordinary HTTP validation, authentication and client
admission layer through `handler_for_host`. No socket is bound by these adapters.

The caller supplies a ready, empty, unclaimed `HostHybridWorker` and matching
`CpuReferenceText`. Both peers must already agree on model/session identity,
precision, token policy, context, slots and layer split. Tokenizer preparation is
not weight identity proof; keep the same externally verified snapshot and the
identity binding used by the [two-process command](cpu-host-command.md).

The peer owner continuously runs `step()`. The coordinator owner runs
`runtime.run()` on one thread; HTTP threads only encode, validate and queue work.
Successful requests release their peer slots before final results, allowing the
next requests to reuse resident weights and transport. Up to the session's four
slots may be active. Prompts are split into bounded activation chunks; ordinary
decode batches retain original slot identity. Prefill is sequential per request,
not interleaved fairly with already active decode.

The supported endpoint is `/v1/completions` with raw text or mapped token IDs,
default greedy sampling and 1–4096 output tokens within the declared context.
Chat templates, tools, custom stops, sampling controls, embeddings, scoring,
reranking, model changes and GPU model-manifest export are rejected. Queue size
and timeout, per-client HTTP authentication/admission, SSE and disconnect cleanup
retain the shared distributed API behavior. This is reference execution, not
production GPU admission or a claim of practical multi-user performance.

Status reports `device: cpu`, `scope: cpu_host_reference`, actual stage ranges,
CPU residency and `official_model_qualified: false`. Timing reports
`source_tier: cpu_reference` and `kernel_backend: cpu-host-reference`; measured
host wall time must not be presented as GPU or network-engine performance. The
reported `session_identity` is the negotiated transport digest, not a newly
computed whole-checkpoint hash. `/health` returns 503 after shutdown or failure.

`runtime.close()` interrupts blocked transport and closes the owned stage. Peer
failure poisons the epoch, wakes pending requests and closes the pair. Retain the
pair/runtime and retry close if stage cleanup fails. Normal owner idle shutdown
may leave an idle healthy pair for an explicit STOP by the caller; no uncertain
state is reused. No restart, remote provisioning or automatic retry occurs.

A request cancelled before prefill dispatch never acquired a peer slot. The
shared distributed owner now tracks that distinction and does not send an invalid
RELEASE for it. Actual TLS/HTTP tests cover output parity against whole CPU
execution, streaming, concurrent slots, reuse, heartbeat, invalid inputs, peer
failure and health after shutdown. Legacy distributed scheduler tests remain in
the regression gate. Hardware and full-model qualification remain open in #51.

The [explicit serving command](cpu-host-command.md#loopback-http-service) owns the
listener, HTTP thread, request owner and peer lifecycle for this adapter. It emits
readiness only after authenticated worker setup and loopback binding.

Distributed output buffering is fixed at eight token events plus one reserved
terminal event per request, independent of the requested output allowance. A full
buffer pauses that slot's decode while other ready slots advance. Final success
still follows acknowledged RELEASE and can be queued without waiting for the
client to consume its last text event. Cancellation and failure do not block on a
full output buffer. When every slot is backpressured, the owner waits briefly and
continues peer heartbeats instead of spinning or advancing model state.

Status exposes `stream_buffer_events: 8`. This is a bound on queued events;
context-bounded token history and cumulative detokenization text remain retained
for each active request, and event byte sizes depend on the tokenizer. It is not
a total process-memory cap. Slow consumers can still occupy their admitted slots
until cancellation or the HTTP socket timeout; no prefill fairness or SLO claim
follows from this change. The scheduling fix is shared with the existing
experimental distributed GPU runtime, whose throughput has not been remeasured.

Distributed admission now captures the HTTP handler's opaque client scope before
handoff and applies the same least-recent client turns as ordinary generation.
Each client's queued requests keep FIFO order; an anonymous caller shares one
FIFO lane. Selection happens under the bounded queue's mutex, so fairness does
not create an extra pending pool or bypass the queue limit. Already cancelled or
expired entries are returned to the existing rejection path without charging a
client turn. Scope objects, never bearer strings or request-body identifiers, are
used as keys. Status reports `client_admission_turns: true`.

This prevents a queued burst from taking every next admission ahead of another
waiting client. It does not reserve active slots, preempt a long prefill, provide
weighted token service or coordinate quotas across separate server processes.
