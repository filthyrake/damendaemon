# Same-host outstanding client reservations

`SharedClientReservations` is an explicit POSIX local-storage backend for
named-client outstanding request counts and declared-output token caps. HTTP
integration is tracked separately in issue #209. Constructing ordinary servers
still creates no reservation store. These counts do not reserve model memory,
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
