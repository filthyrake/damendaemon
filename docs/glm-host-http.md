# Prepared native GLM host HTTP

`HostGlmReferenceRuntime` in `dllm.glm_host_serving` explicitly owns one prepared
`HostGlmGenerator`. Use `handler_for_reference` and the existing bounded HTTP
server/authentication policy. Peer workers must continue stepping externally.
Generic `ReferenceRuntime` still rejects a host runner, and the prepared host
owner must be healthy, idle and unclaimed. The explicit topology command can launch this owner in native `serve` mode;
see [glm-host-command.md](glm-host-command.md). There is no queue here; concurrent
generation receives the existing busy response.

The ordinary reference chat JSON/SSE routes retain their native reasoning,
function declaration, history, usage and default-greedy restrictions. No tools
are executed. Status and timing identify `execution: cpu_host_topology`, layer
ranges, source and topology identities, whether source bytes were verified, and
whether the epoch is terminal. Scope remains CPU reference and official model
qualification remains false. HTTP batches are not admitted by this adapter.

Healthy requests wait for remote slot release and parser cleanup, then permit
sequential reuse. Abandonment after dispatch, active cancellation and remote
failure retire the epoch and prevent new requests. Readiness is unhealthy after
observed transport failure or retained cleanup failure. It describes observed
session state; it is not a fresh peer liveness probe.

`close()` stops admission and interrupts host transport I/O before waiting for
active model ownership. It raises `BusyError` while an iterator still owns HTTP
execution; retain the runtime and retry after that iterator exits. Failed stage,
transport or parser cleanup also retains the owner for retry. A terminal epoch
requires a newly prepared session, even after cleanup succeeds. Explicit close
uses transport shutdown rather than a graceful STOP exchange.

Tests run actual local mutual-TLS stages behind authenticated loopback HTTP.
JSON text/usage matches an independent whole-model CPU control; SSE and native
tools execute actual packed stage math (controlled output selection is used for
tool-wire tests). Negative checks cover unchanged generic admission, malformed
requests before dispatch, busy ownership, abandonment, remote failure, retained
cleanup/readiness, and blocked-receive interruption. This does not qualify
physical networks, official checkpoints, GPUs, practical context or throughput.
