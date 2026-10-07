# Remote quota HTTP admission

Operator bootstrap, enrollment, frontend opt-in and explicit recovery are documented in [quota commands](quota-commands.md).

`RemoteQuotaReservations` adapts the mutually authenticated quota client to the
existing named-client HTTP lease boundary. Embedders can construct the adapter
and pass it as `reservation_store` to `handler_for` or `ClientAdmission`.
Commands opt in with `DLLM_REMOTE_QUOTA_CONFIG` and the owner's runtime credential;
unset configuration preserves local admission without a network connection.

The authority's authenticated `policy` RPC returns only that owner's allowed
client names and request/output caps. Before admission, the adapter verifies all
configured named-client caps match and reads owner status. A retired owner,
policy mismatch, exhausted sequence or prior unresolved claims prevents startup.
Do not infer that previous work stopped because a frontend or authority restarted.
Recovery must resolve that work explicitly before returning capacity.

Use one stable authority owner per frontend controller. Mutations serialize under
the controller lock. Each new mutation gets a unique nonce and increasing
sequence; its exact immutable request is retained until a verified receipt arrives.
This prevents separately created requests with identical counts/sequence from
sharing a charge. A stale or conflicting writer fails closed. The adapter never
re-sequences an uncertain command automatically.

A successful reserve returns an owner-bound handle. A failed/unknown reserve
exposes its candidate as `SharedReservationUnavailable.claim` and retains it in
`store.claims`; `store.pending` retains the exact operation. To end that failed
lease, call `release(candidate)`. It first resolves the original saved reserve.
Only a verified acceptance permits a subsequent release. A verified rejection
removes the uncharged candidate. An unresolved/conflicting receipt retains the
candidate, even when its sequence-derived ID resembles another controller's
claim. It must never become permission to release that other claim.

HTTP admission retains failed handles through its existing `pending_releases`
path. `retry_releases()` retries ended leases in order, resolving the earliest
uncertain operation first. Live request handles cannot be discarded by close.
Failed release retries use the same saved request; closing a controller may
complete an already requested release, but it does not release live work or retire
the authority owner. No transport socket remains open between calls.

Named-client request claims span response cleanup. Declared-output claims span
generator cleanup. Model permissions, opaque client scopes, shared/admin bypass,
and rolling-rate charging remain unchanged. Returning capacity does not refund a
rate charge. Capacity rejection is 429; unavailable accounting is redacted 503.
SSE cleanup failure suppresses `[DONE]`; a failed request release after a complete
response closes the connection without sending a second HTTP response.

Local tests use a real TLS authority and multiple HTTP frontends, including quota
contention, client isolation, authority restart with active leases, lost accepted
or rejected replies, lost releases, model failure, disconnected streams, rate
preservation, policy mismatch and conflicting controllers. These are accounting
and lifecycle proofs. They do not establish remote workload fencing, automatic
dead-host reclamation, GPU memory reservation, real-network throughput or an SLO.
