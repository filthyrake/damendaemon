# Owned CPU host-worker construction

`damendaemon.cpu_host_worker_build.build_cpu_host_worker` composes stage-local checkpoint
loading, a prepared CPU stage and host-worker protocol ownership:

```python
from damendaemon.cpu_host_worker_build import build_cpu_host_worker

worker = build_cpu_host_worker(
    snapshot, start, end, fresh_transport,
    epoch=1, max_context=512, max_batch=2, quantization='q4_0',
    max_weight_bytes=256 * 1024**2,
    loader_workspace_bytes=64 * 1024**2,
    max_cache_bytes=256 * 1024**2,
    projection_workspace_bytes=8 * 1024**2,
    max_workspace_bytes=64 * 1024**2,
    max_output_bytes=64 * 1024**2,
)
```

The caller supplies a fresh `HostTensorChannel` with an explicit model/session
identity and rank. Rank 0 must own the embedding and a leading, non-final layer
range; rank 1 owns the remaining final range and output head. The returned worker
is fresh and does not handshake during construction. Run both peers' explicit
handshakes before submitting commands. Setup binds their identity, configuration,
quantization, context, slots and complete contiguous layer coverage.

Authenticate sockets through [host endpoints](host-endpoints.md) before creating
transports on real links. The factory does not change the transport's trust
policy, create sockets or threads, bind a listener, download a model or deploy a
worker. Trusted plaintext loopback transports remain possible for existing tests.
An explicit `context_limit` is required above the protocol's legacy 512-token
limit; larger metadata admission is not context qualification.

Control validation happens before transport ownership. Once claimed, the factory
owns that transport through selected payload loading, CPU stage construction and
worker construction. Success transfers ownership to the returned worker; call its
`close()` when finished. A duplicate claim is rejected.

Failures raise `CpuHostWorkerBuildError`, preserving the primary failure in its
message and closing the transport/channel and any partially initialized stage.
If its `closed` flag is false, retain the exception and retry its `close()`; partial
storage remains reachable until cleanup succeeds. Original loader tracebacks and
unneeded source tensors are not retained. Construction never starts inference or
waits for peer protocol traffic. Weight, loader scratch, stage/cache/workspace and
frame limits remain explicit and independently checked.

Actual local TLS peers load their own Q4/Q8 checkpoint ranges and match whole CPU
pipeline sampled outputs through chunked prefill, ragged batched prefill, reordered
decode, release/reuse and stop. Setup mismatch, invalid control, failed loads,
partial stage construction and retryable cleanup are tested. These CPU fixtures
do not qualify trained-model quality, practical context, real-network throughput,
hardware, continuous service or production concurrency.

## Mapped output policy

Pass `mapped_vocab_size` and optional `excluded_ids` to the factory or prepared
`HostHybridWorker` constructor when the tokenizer maps only a prefix of the model
head. Use a validated contiguous tokenizer map (for example `CpuReferenceText`),
not a count from a vocabulary containing holes. Omitting the size preserves the
explicit all-head-rows-mapped contract; the worker cannot infer a tokenizer it has
not received.

Configured media IDs are automatically excluded, including with default options.
Additional exclusions must be valid model IDs. Reserved media IDs outside the
head remain naturally unreachable. At least one mapped output must remain.
Both peers bind this immutable policy into setup identity and reject mismatches
before inference. Input rejects excluded/unmapped IDs before command dispatch;
output masks them before greedy sampling. A policy-violating peer result poisons
and closes the epoch instead of returning success. Such failure is not a durable
distributed rollback; a peer may already have advanced its local cursor.

This fixes a reproduced asymmetry: a configured media ID rejected on input was
previously returned when its output logit was highest. CPU loopback regressions
cover high media/padded logits through chunked/batched prefill and reordered
decode, default media exclusion, bad-policy ownership and mismatched peers.
