# Explicit quota authority commands

The authority accounts for outstanding requests and declared output tokens. It
does not fence remote execution, measure GPU memory, or reclaim work on a timer.
The default HTTP frontend remains process-local. These commands create no
listener unless `serve` is explicitly run. Local tests qualify command behavior,
not production networks, practical model concurrency or performance.

Use owned, mode-0600, single-link regular JSON files with absolute paths; symlink
inputs, duplicate/unknown fields and inputs above 64 KiB are rejected. Ledger
storage requires a private POSIX directory. TLS uses the existing TLS 1.3 mutual
authentication, CA verification, exact certificate pins and `dllm-quota/1` ALPN.
Private keys remain external files. Encrypted keys may use
`DLLM_REMOTE_QUOTA_TLS_PASSWORD`; credentials never belong in repository files.

## Bootstrap and enrollment

The policy file contains `path` (absolute ledger filename) and `clients`: a list
of objects with `name`, `max_inflight` (1–128), and
`max_reserved_output_tokens` (1–2147483647, or null for no output cap). Names and
caps must match the HTTP frontend's named client policy. Do not include API keys.

```sh
python -m scripts.serve_quota_authority init --policy /private/quota/policy.json
python -m scripts.serve_quota_authority register \
  --policy /private/quota/policy.json --owner "$QUOTA_OWNER_ID" --client alice
```

Create a stable owner ID from 16 random bytes encoded as 32 lowercase hex
characters. Supply an independent 32-byte random owner credential as 64 lowercase
hex characters in `DLLM_REMOTE_QUOTA_OWNER_KEY` through the secret manager or
process environment. Enrollment stores only its hash; stdout contains public
ledger and owner IDs. Repeating enrollment with identical policy/credential is
idempotent. Retired owners cannot be resurrected. Identities, including retired
tombstones, are bounded to 128; do not allocate an owner per request.

Only `init` creates storage. Ordinary startup rejects missing/replaced state.
Preserve the ledger and its `.identity` anchor together; copying/replacing a DB
file is not supported recovery. Retain the issued ledger identity for clients.

## Serve and opt in a frontend

Listener JSON requires `address` (literal IPv4/IPv6), `port` (0–65535), absolute
`ca_file`, `cert_file`, `key_file`, and `client_pins` (1–128 distinct certificate
SHA-256 DER pins as lowercase hex). Optional `max_connections` is 1–128 (default
16); `timeout` is greater than zero and at most 30 seconds (default 5).

```sh
python -m scripts.serve_quota_authority serve \
  --policy /private/quota/policy.json --listen /private/quota/listener.json
```

`quota_ready` JSON reports the bound address/port and ledger identity. SIGINT or
SIGTERM closes owned connections/workers and emits `stopped` after successful
cleanup. Busy cleanup is retried within bounded close attempts; failure exits
nonzero. Neither successful shutdown nor a crash returns remote quota.

Frontend JSON requires literal `address`, `port` (1–65535), `server_hostname`
(the certificate name), `ledger`, `owner`, `server_pin`, and absolute `ca_file`,
`cert_file`, `key_file`; optional `timeout` has the same bounds. The owner and
ledger IDs are 32 lowercase hex characters and the pin is 64. Set
`DLLM_REMOTE_QUOTA_CONFIG` to that private file and supply the owner credential
through `DLLM_REMOTE_QUOTA_OWNER_KEY`. Existing named-client configuration stays
in `DLLM_CLIENT_KEYS`; administrator/shared API keys retain their existing
policy. Do not give those keys to tenants who must be capped.

All frontends using `ClientAdmission` opt in through this environment setting,
including the CPU reference command. Startup verifies authenticated authority
policy and empty owner claims. Simultaneous `DLLM_CLIENT_RESERVATION_DB` and remote
configuration is rejected; there is no fallback when the authority is unavailable.
Explicit library injection takes precedence over environment selection. Rolling
rate quotas remain independent; their optional SQLite store is same-host only.

Use exactly one controller per owner, including during recovery. Do not start a
replacement while prior work can still run. Remote errors keep claims charged
and block further admission until ended-lease cleanup is resolved. The CPU
command waits for accepted HTTP threads and their quota releases on shutdown;
slow/incomplete request bodies can hold it until the existing body socket timeout.
An unresolved cleanup error prevents a successful `stopped` report.

## Inspect and explicitly recover ended work

```sh
python -m scripts.serve_quota_authority status --config /private/quota/frontend.json
python -m scripts.serve_quota_authority claims --config /private/quota/frontend.json --limit 128
python -m scripts.serve_quota_authority recover \
  --config /private/quota/frontend.json --operation /private/quota/release.json
```

Inspection is owner-authenticated. Claims are paginated; pass the returned
`after` cursor with `--after` for the next page. Inspection does not release them.
Establish that the old frontend and its execution have ended before recovery.
An unreachable frontend, elapsed time, or an authority restart is insufficient.

The operation file contains exact `ledger`, `owner`, `sequence` (next authority
sequence), a fresh `operation_id` (16 random bytes as 32 lowercase hex), and
`command`: either `{"op":"release","claim":"OWNER:SEQUENCE"}` or
`{"op":"retire"}`. Fill in actual identity values; this example is not an
executable operation. Save the file **before** invoking recovery. It contains no
credential. Retry precisely the same file after an unknown result. Never choose
a new sequence/nonce merely because a reply was lost. A changed retry conflicts;
only the latest exact operation can be replayed. Retirement requires zero claims.
Read the receipt and status before proceeding to the next operation. Rejected
mutations also consume a sequence. A verified negative mutation receipt returns exit 1;
configuration, transport or protocol failure returns exit 2.

The CLI deliberately has no reserve/resize, remote enrollment or automatic stale
owner command. The [ledger](remote-reservation-ledger.md),
[transport](remote-quota-transport.md) and [admission](remote-quota-admission.md)
documents define the underlying ownership and uncertainty contracts.
