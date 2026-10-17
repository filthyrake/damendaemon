# Same-host outstanding client reservations

`SharedClientReservations` is an explicit POSIX local-storage backend for
named-client outstanding request counts and declared-output token caps. HTTP
admission opts in with `DAMENDAEMON_CLIENT_RESERVATION_DB=/private/path/claims.sqlite`.
Without this variable, ordinary servers still create no reservation store. These counts do not reserve model memory,
establish cross-host quotas or demonstrate an inference SLO.

Pass an absolute SQLite path in a private, owned local directory and the named
`ClientKey` policies. All processes using a client name must agree on its
`max_inflight` and `max_reserved_output_tokens`; keys and scheduler weights may
differ. API credentials and request content are never persisted.

`reserve(client, outputs=0, requests=1)` atomically claims capacity.
`requests=0` with positive output reserves only declared output, for a separate
HTTP output lease. `resize(claim, outputs)` changes the output reservation
atomically. `release(claim)` is idempotent for the exact owner-issued handle.
Another owner cannot resize or release it. Release all handles before `close()`;
closing a live owner refuses to discard its reservations.

Each owner holds one OS lock in a fixed, private 128-file sibling directory
named `<database>.owners`. Transactions reclaim another owner's records only
after acquiring its lock nonblockingly and checking the file identity. There
is no lease expiry, PID lookup or timeout-based reclamation. A surviving process
keeps its reservations while other processes restart; process termination
releases its OS lock. Inherited/forked store objects are rejected: construct
owners after forking. Network filesystems are unsupported.

The store binds database, directory and lock-file identities. Missing, replaced,
unsafe or corrupt storage fails closed. Do not delete, copy, replace or restore
these files while owners may exist. The bounded pool has 128 owners, 128 client
policies, 32,768 claims, a 32 MiB database ceiling and one-second SQLite lock
waiting. The rollback journal may also occupy up to database size. Filesystem
calls and local thread scheduling are not hard realtime. Pool/policy exhaustion
requires retiring owners or an explicit offline migration, not automatic eviction.

A failed mutation may have committed. `SharedReservationUnavailable.store`
retains the owner; failed reserves retain their handle in `store.claims`.
The owner refuses new reservations while an operation is uncertain. Retry
release for unresolved handles and retry close after repairing storage. Do not
drop the owner, unlock its file or treat an error as returned capacity. Failed
construction likewise retains an owner on the exception if cleanup needs retry.

Local tests cover process contention and crash recovery, surviving owners,
request/output caps, output resizing, commit uncertainty, retry, ownership and
pool bounds, storage replacement, corruption and unsafe permissions. They do
not qualify distributed performance or GPU capacity.

## HTTP ownership and recovery

Every participating server must use the same absolute local database and matching
named-client caps. Request claims span the entire HTTP handler, including invalid
body rejection and response cleanup. Separate output claims cover declared output
through generator cleanup. Shared/admin keys retain their existing bypass policy;
client keys, model allowlists and opaque scheduling identities are unchanged.
Use a separate `DAMENDAEMON_CLIENT_RATE_DB` when rolling rates must also be shared:
returning outstanding capacity never refunds an accepted rate charge.

A cap rejection returns 429. Unavailable accounting returns a redacted 503 before
response headers; a failed output release during SSE emits an error and suppresses
`[DONE]`. If request cleanup fails after a complete response, the connection closes
without a second HTTP response. The uncertain claim remains charged, and that
admission owner rejects new named-client work until cleanup succeeds.

`BoundedHTTPServer.server_close()` closes the reservation owner after leases have
ended. It refuses active leases and retains failed cleanup for a later close
attempt. Drain/stop generation and end HTTP requests before closing. Bind failure
also cleans up an unused owner. Embedders using another HTTP server class must
explicitly own and close `handler.client_admission` after draining their server.
Do not share one admission store object between independent handlers.

For an embedded server, `handler.client_admission.retry_releases()` retries only
ended or failed leases; it never frees live request claims. A controlled process
restart also permits a new owner to reclaim the old unlocked slot after exit.
Storage identity failures require restoring the original intact storage first;
restarting is not permission to discard a live or unknown owner. No network admin
endpoint can override these ownership checks.
