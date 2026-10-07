# Owned CPU host-worker construction

`dllm.cpu_host_worker_build.build_cpu_host_worker` composes stage-local checkpoint
loading, a prepared CPU stage and host-worker protocol ownership:

```python
from dllm.cpu_host_worker_build import build_cpu_host_worker

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
