# #536 gate-4 — firewalld opened (scoped) and DIRECT cross-host transfer measured

Date: 2026-10-19. Supersedes the SSH-tunnel bound in `gate4-transfer-controls.md`
as the transport-planning number. No GPU execution, no NCCL qualification.

## Firewall change (labnode3, firewalld permanent + runtime)

Rich rules in zone `public`, scoped to the desktop's address only:

```
rule family="ipv4" source address="10.40.100.30" port port="9321" protocol="tcp" accept
rule family="ipv4" source address="10.40.100.30" port port="29500-29510" protocol="tcp" accept
```

Scope verified from the desktop: 9321 and 29500 OPEN; 50051 (listener, no
rule) and 12345 (no listener) still blocked. The 29500-29510 range is the
torchrun rendezvous band for the #514/#515-fixed launcher; NCCL's own socket
listeners bind ephemeral ports on the same interface, so a broader NCCL range
(or NCCL_SOCKET_IFNAME pinning) remains a runner-time decision.

## Direct (non-tunneled) measurements, desktop → labnode3

Round-trip echo (`gate4_transfer_probe.py`, 50 repeats, same-connection):

| payload | median RTT | max |
|---|---|---|
| 10 KiB (decode token) | **0.50 ms** | 0.95 ms |
| 1.27 MiB (chunk 127) | 5.21 ms | 7.07 ms |
| 1 MiB block | 4.01 ms | 4.84 ms |

One-way bulk: **2.00 GiB in 7.29 s = 281 MiB/s (2.36 Gbit/s)** single TCP
stream. (Receiver was a docker `--net=host` container after repeated sshd
session-kill issues with setsid children — the transfer path is plain TCP on
the host NIC either way.)

## Gate-4 admission math with the real numbers

- Decode token hop: 0.50 ms RTT ⇒ two serialized boundaries ≈ **1.0 ms/token
  floor** ⇒ up to ~1000 tok/s transport-bound — decode transport is a
  non-issue at batch 1; even 20 tok/s/user at 4 users needs 80 hops/s ≈
  4% duty.
- Prefill chunk-127 hop: 5.2 ms per boundary ≈ 10.4 ms per chunk, and at
  281 MiB/s the 1.27 MiB payload itself is bandwidth-trivial. The gate-3
  8K ramp was 64 chunks ⇒ ~0.7 s added transfer across both boundaries —
  negligible against the thermal profile that gated it.
- The 2.36 Gbit/s single-stream ceiling leaves NCCL headroom via its own
  multi-stream/pipelined protocol; bulk is not the binding constraint at
  these payload sizes.

## Conclusion for the runner

Transport no longer gates the three-rank qualification: RTT and bandwidth
are both an order of magnitude inside the budget. The runner should (a) use
torchrun/NCCL directly with the 29500-29510 rendezvous, (b) pipeline prefill
chunks as already planned, (c) record per-token host-transfer cost from the
NCCL path — never from userspace relays. The SSH-tunnel numbers in
`gate4-transfer-controls.md` are retained as the documented routing bound
they were.

## Reproduction

- Firewall: `firewall-cmd --zone public --add-rich-rule=... --permanent` +
  runtime (exact rules above; verified by readback).
- RTT: probe server on labnode3 (:9321), client `--host 10.30.100.3`.
- Bulk: 1 MiB × 2048 sender; docker-hosted receiver counted 2147483648 bytes.
