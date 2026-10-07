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
