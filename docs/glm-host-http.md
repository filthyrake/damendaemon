# Prepared native GLM host HTTP

`HostGlmReferenceRuntime` in `dllm.glm_host_serving` explicitly owns one prepared
`HostGlmGenerator`. Use `handler_for_reference` and the existing bounded HTTP
server/authentication policy. Peer workers must continue stepping externally.
Generic `ReferenceRuntime` still rejects a host runner, and the prepared host
owner must be healthy, idle and unclaimed. The explicit topology command can launch this owner in native `serve` mode;
see [glm-host-command.md](glm-host-command.md). This base owner has no queue; concurrent
generation receives the existing busy response. The explicit
`QueuedHostGlmReferenceRuntime` composes bounded serialized admission with this
host owner, preserving terminal-epoch behavior rather than local-model reuse.

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

The native GLM/Flash/DeepSeek command explicitly uses non-daemon HTTP request
threads. Shutdown interrupts runtime/peer waits before joining these threads,
then retries runtime cleanup; successful STOP output follows HTTP quota release.
An incomplete HTTP body keeps ownership until the existing read timeout or body
completion. Authority loss can leave remote claims charged and makes cleanup
fail; it must not produce a graceful-stop report. Active generation cancellation
still retires the epoch, while idle/drained shutdown requires acknowledged peer
STOP before reporting graceful completion.

Tests run actual local mutual-TLS stages behind authenticated loopback HTTP.
JSON text/usage matches an independent whole-model CPU control; SSE and native
tools execute actual packed stage math (controlled output selection is used for
tool-wire tests). Negative checks cover unchanged generic admission, malformed
requests before dispatch, busy ownership, abandonment, remote failure, retained
cleanup/readiness, and blocked-receive interruption. This does not qualify
physical networks, official checkpoints, GPUs, practical context or throughput.


## Queued host ownership

The queued specialization uses the existing CPU reference queue through a narrow
owned-generation hook. Up to 64 pending requests each have eight buffered data
events plus reserved terminal space, a bounded JSON input snapshot, timeout and
client admission scope. Exactly one thread generates. Invalid requests and
undispatched queue expiry/cancellation leave the healthy host alone. Active
abandonment, transport failure and cleanup failure retire the epoch and fail
waiting requests. A failed epoch never starts the next queued generation.

Drain stops admission and finishes accepted requests. Scheduler status reports
pending/owned counts and a single-generation batch capacity. Optional command
stop-after-drain waits for those counts to reach zero before acknowledged STOP.
Close interrupts host receives before joining the queue thread; retained stage
or parser cleanup remains observable and retryable. Tests cover actual concurrent
HTTP/control agreement, client turn order, input snapshot isolation, queue
pressure/expiry, active abandonment, remote failure, blocked receives, retained
cleanup and subprocess drain while admitted requests are still outstanding.
