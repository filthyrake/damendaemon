# Bounded authenticated host endpoints

`dllm.host_endpoint` adds explicit TCP setup around the existing host TLS policy.
It does not bind an address, discover hosts, provision certificates, launch a
worker or register a service. Tests use local loopback only.

`connect_peer(literal_ip, port, client_context, peer_pin=server_pin,
server_hostname=certificate_hostname, timeout=10, cancelled=None)` returns an
owned, authenticated blocking TLS socket. The routing address must be a literal
IPv4/IPv6 address without a zone identifier; the separately supplied hostname is
verified against the server certificate. There is no DNS lookup or retry. One
absolute deadline covers connection and TLS authentication, including certificate
pin and ALPN checks. All partially opened sockets close on failure.

`HostPeerListener(bound_tcp_socket, server_context, peer_pin=client_pin,
backlog=4)` validates the caller's socket, then starts listening with a backlog of
1..16. It accepts an already-listening socket too, explicitly resetting its
backlog. The caller chooses and binds the address; unbound or connected sockets
reject. This avoids relying on a listening-status socket option that is not
readable on every supported host. Failed input validation leaves ownership with
the caller; a successful constructor owns the socket.

`listener.accept(timeout=10, cancelled=None)` permits one pending accept. Its one
deadline includes waiting for TCP plus TLS. A rejected client closes without
closing the listener, which can accept a later client. Successfully returned peers
become caller-owned; closing the listener does not close them. Pass each returned
peer to a `HostTensorChannel`/`HostStageChannel` and still perform the model/session
identity handshake. TLS authenticates transport; it does not replace protocol
admission or authorize arbitrary model operations.

`listener.close()` is idempotent and interrupts a pending accept or stalled TLS
handshake. TLS and socket waits poll cancellation at intervals of at most 50 ms;
this is a software polling bound, not a scheduling-latency guarantee. Timeouts
must be finite and within (0, 30] seconds. Cancellation callbacks must be prompt
and nonblocking. Cancelled operations raise `HostTlsCancelled`; timeout raises
`TimeoutError`. Failed operations never transfer a socket. There is no plaintext
fallback. Keep TLS contexts immutable while in use and do not share owned sockets
with other readers or writers.

The shared `host_tls.wrap_peer` now uses cancellable nonblocking handshakes and
restores the supplied socket's prior timeout on success. Existing requirements
remain TLS 1.3, mutual trusted certificates, exact SHA-256 DER peer pins, client
hostname verification and `dllm-host/1` ALPN. Configuration errors before ownership
leave the raw socket with its caller. The endpoint helpers close their own raw
sockets even if that policy validation fails later.

Loopback tests pass actual tensor frames after bootstrap, preserve transferred
peers after listener closure, exercise wrong-pin rejection/reuse, refused TCP,
combined deadlines, stalled TLS cancellation, close races and concurrent accept
rejection. They do not qualify real-network throughput, hardware, deployment,
certificate rotation or host-worker provisioning.
