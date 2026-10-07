# Planned CPU host stages

`HostTopology` defines an immutable, explicit contract for 2–8 contiguous,
nonempty layer ranges covering a prepared text-only Qwen3.5 CPU model. It binds
source/session identity, configuration, FP32 computation, none/Q4_0/Q8_0 weight
policy, mapped/excluded token IDs, context/slots, prefill chunk rows and frame
budget. Configuration is copied into at most 1 MiB of canonical JSON; subsequent
caller mutations cannot change it. Context remains bounded by both model
positions and the explicit admission limit. Frame admission covers the configured
chunk and full decode batch, not weight/cache/workspace residency.

`stage_identity(index)` derives a separate identity for every global stage,
including whether it returns intermediate activations or final sampled tokens.
`PlannedHostStageChannel` is an explicit alternative to the unchanged legacy
`HostStageChannel`: each remote stage has a coordinator-to-worker link with local
ranks 0/1, while its global index is 1..N-1. The transport identity, frame cap,
context and slots must match the topology before ownership transfers. Setup also
exchanges the stage range, topology size, width and result role. Owners must
use the existing mutual-TLS/pinned connection setup before constructing the
transport; topology identity is not authentication.

Intermediate stages complete execution with contiguous, finite CPU FP32 tensors
of shape `(command.rows, hidden_size)`. The coordinator validates the returned
frame before completion acknowledgement and cursor commit. Final stages use
sampled CPU int64 IDs, now checked against the planned mapped/excluded policy
before commit. Result-mode confusion, excessive prefill chunks, wrong frame
shape/dtype, nonfinite values and invalid IDs abort the epoch. Release, ping,
stop, slot generations and frame deadlines retain the existing control semantics.

`HostPipelineCoordinator` owns the prepared first CPU stage and ordered links
for every remaining range. `HostPipelineWorker` owns exactly one intermediate or
final stage. Constructors validate fresh, unclaimed resources against the full
plan before claiming them. Handshake every participant before submitting work.
All links admit each command before first-stage computation; intermediate results
are routed through the coordinator and only the final stage samples mapped,
non-excluded greedy tokens. This synchronous star routing favors explicit
ownership and failure handling; it makes no throughput claim.

The same owner supports packed/chunked prefill, reordered/shrinking decode,
release/reuse, ping and acknowledged STOP. `HostCompletion` supplies ragged
request generation, EOS and release-before-success. Requests exceeding planned
chunk rows fail before dispatch. Any uncertain result discards every local link
and local stage; peers observe closed connections. Close attempts every resource,
preserves the primary execution error and retains failed cleanup for retry.
Acknowledgements are not durable distributed transactions: an earlier stage can
commit before a later stage fails, so reconnect/resume/replay is unsupported.

Actual loopback mutual-TLS tests compare three- and four-stage FP32/Q4/Q8
execution against independently expanded whole-model logits and cache state,
and cover cancellation, admission failure and intermediate/final worker loss.
These are prepared-stage APIs: snapshot construction, command bootstrap and HTTP
integration still need explicit topology adapters. Source identity remains a
caller assertion unless a separate verified snapshot path supplies it. No GPU,
real-network, model-quality or performance qualification is implied.
