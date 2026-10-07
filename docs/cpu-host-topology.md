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

This is a protocol prerequisite, not yet multi-stage model execution. It does
not route activations, construct model stages or deploy workers. The next owner
must compose all links and discard the whole topology on any uncertain result:
acknowledgements are not durable distributed transactions, and peers can differ
briefly in local commit position after connection loss. Actual loopback TLS tests
cover typed results, chunks, decode, release/reuse, corruption and cancellation;
no GPU, real-network, model-quality or performance qualification is implied.
