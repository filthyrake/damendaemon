# Durable remote-owner reservation ledger

`RemoteReservationLedger` is a local accounting foundation for cross-host
admission. It does not start a listener, change HTTP defaults, fence remote
workloads, reserve GPU memory or establish an inference SLO. The authenticated
network authority/client integration is a separate gate.

Bootstrap a new absolute private local SQLite path explicitly with `create=True`.
Normal construction requires existing storage; it never silently recreates a
missing ledger. A private sibling `.identity` file binds the database inode and
ledger identity. Existing files cannot be overwritten by bootstrap. Keep both
files intact: copying/restoring/replacing live authority storage is unsupported
and can invalidate the identity. The configured client request/output caps must
match across authority handles. API client keys are never persisted.

Privileged local code registers a stable 32-character lowercase hexadecimal
owner ID, a random 32-byte owner credential and a bounded client-name allowlist.
Only a SHA-256 credential hash is stored. Registration retries require the same
credential and allowlist. These owner credentials are distinct from end-user API
keys. A future network layer must authorize registration separately; possessing
an owner's credential must not grant enrollment authority.

Each owner serializes commands with integer sequences starting at one:

- `reserve`: exact fields `op`, `client`, `requests` (0 or 1), `outputs` (0 through
  2,147,483,647), with a nonzero combined claim. Claim IDs derive from owner/sequence.
- `resize`: exact fields `op`, `claim`, `outputs`. Empty output-only claims must be
  released explicitly.
- `release`: exact fields `op`, `claim`; removing an already absent owner-bound
  claim is harmless.
- `retire`: exact field `op`; retirement refuses outstanding claims.

The last command's digest and result are committed with its effect. Retry the
same sequence and exact payload after a lost/uncertain reply. Network callers
also supply a stable per-operation nonce, included in the digest; a distinct
operation with identical counts still conflicts at the same sequence. Changed retries,
stale commands and skipped sequences cannot mutate state. Quota rejection is
also a durable result: use a new sequence for a later attempt. Do not issue the
next command until the prior result is known. `status` returns the authenticated
owner's sequence, latest receipt and outstanding count; `list_claims` provides
owner-only pages of at most 128 entries for recovery. Neither call releases work.

An authority close, restart, process death, socket loss or elapsed interval
**never returns remote capacity**. Authenticated release is required. An abandoned
owner can therefore consume capacity indefinitely; this is deliberate until a
separate workload-fencing/recovery contract proves safe reclamation. The local
same-host OS-lock backend cannot supply that remote proof.

Storage is bounded to 128 client policies, 128 stable registered owner identities
(including retired tombstones), 32,768 live claims and a 32 MiB SQLite database.
The rollback journal can occupy up to database size. Tombstones prevent delayed
registration from resurrecting retired identities; there is no automatic owner
eviction. Use deployment identities rather than one owner per request. Exhaustion
requires an explicit offline migration after resolving all owners, not deletion
of live state. Sequence numbers are bounded to signed 64-bit positive integers.
SQLite lock waiting is one second; filesystem calls are not hard realtime.

Private regular storage, schema/identity/ownership/quota invariants and foreign
keys are checked; unknown outcomes fail closed and require retry of the same
operation. Local POSIX filesystems are required. Network filesystems, hostile
privileged writers and rollback to an older valid database are outside this
storage contract. Local fixtures prove accounting and failure behavior only,
not live distributed inference or useful throughput.
