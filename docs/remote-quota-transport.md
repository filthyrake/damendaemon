# Bounded remote quota transport

`RemoteQuotaServer` and `RemoteQuotaClient` expose the existing
[durable remote-owner ledger](remote-reservation-ledger.md) over one-request
mutual-TLS connections. Construction is explicit; no inference server, command
or import starts an authority automatically. Only local test listeners have been
exercised. This is accounting infrastructure, not remote workload fencing,
automatic dead-host reclamation or distributed inference qualification.

Both endpoints require TLS 1.3, trusted client/server certificates, exact DER
certificate pins and the distinct `dllm-quota/1` application protocol. Build
contexts using `client_context(..., alpn=QUOTA_ALPN)` and the corresponding server
function in `dllm.host_tls`. Clients pin one authority certificate and validate
its hostname. Authorities explicitly allow at most 128 distinct client pins.
Existing tensor-worker TLS retains `dllm-host/1` by default. Keep contexts
immutable after setup. No certificate/private key is embedded in the repository;
TLS tests generate temporary credentials.

Connect/listen addresses are explicit IP literals plus ports; certificate
hostnames are separate. This avoids unbounded DNS resolution inside a socket
operation. Clients also pin the expected 32-hex ledger identity. A replaced or
incorrect ledger cannot be silently accepted merely because its certificate
matches. Peer and ledger identity failures never trigger plaintext fallback or
identity re-enrollment.

An authority exposes only `execute`, `status` and bounded `list` operations for
already registered owners. There is no registration, credential rotation,
forced release or administrative reclamation RPC. Owner registration and
allowlists remain privileged local operations. Each request carries its owner
credential inside TLS; credentials and request bodies are not logged.

`client.request(method, **options)` creates an immutable request snapshot, whose
representation hides the credential. `client.call(snapshot)` opens one TLS
connection, sends it and validates the complete response. Each new mutation
snapshot gets a random operation identity included in the durable digest, so
independent callers cannot accidentally share a charge by choosing the same
sequence and counts. An unknown/lost result
raises `RemoteQuotaUnavailable`; retry **the same snapshot**. Never create a new
sequence merely because the prior response was lost. Rejection responses and
ledger command receipts remain distinct. Client validation rejects foreign claims,
wrong sequences, malformed receipts/status/pages and mismatched authority identity.

Wire format is a four-byte unsigned network-order length followed by bounded
UTF-8 JSON. Requests are at most 4 KiB and replies at most 64 KiB. Duplicate fields,
nonfinite JSON, unknown methods/fields, unsupported protocol versions and invalid
framing fail closed. Each connection handles exactly one request. The authority
has 1–128 connection workers (default 16); excess connections close before TLS.
Handshake/socket waits share a configurable 0–30 second positive deadline
(default five seconds). SQLite independently bounds lock waiting to one second;
filesystem calls and OS scheduling are not hard realtime. A deadline can leave a
committed operation with no visible reply, which is why receipts are durable.

After construction, start `server.serve_forever()` in one owned thread. Successful
construction transfers ownership of the ledger handle and listener to the server.
Failed construction leaves the caller's ledger usable. `server.close()` stops
accepting, interrupts owned sockets and waits at most two seconds for threads.
If cleanup remains active it raises and retains those owners for retry. Only
completed local cleanup closes the ledger handle. Neither shutdown nor a client
disconnect releases any remote claim; authority restart preserves charges.

Local TLS fixtures cover exact retries after lost committed replies, restart,
wrong pins/credentials/ledger/protocol, certificate rejection, malformed frames,
connection saturation, deadlines, bounded response validation and retryable
shutdown. HTTP admission adaptation and standalone authority operation are
following gates.
