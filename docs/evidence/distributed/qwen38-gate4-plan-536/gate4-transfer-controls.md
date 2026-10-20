# #536 gate-4 cycle-1 feasibility controls — boundary sizes + measured cross-host RTT

Date: 2026-10-19. CPU/network only; no GPU execution, no NCCL qualification.

## Boundary-size table (from the #551 plan JSONs, `plan_q4_split.py` output)

| context | split layers (A30/A2/3090) | decode bytes per token per boundary | chunk-127 bytes per boundary | max util |
|---|---|---|---|---|
| 2048 | 0–24 / 24–44 / 44–64 | 10,240 B (1×5120×FP16) | 1.27 MiB | 36% |
| 8192 | 0–24 / 24–43 / 43–64 | 10,240 B | 1.27 MiB | 38% |
| 32768 | 0–25 / 25–43 / 43–64 | 10,240 B | 1.27 MiB | 45% |

Two host boundaries per token (rank0→rank1, rank1→rank2) plus an 8-byte
sampled-token return. Even at 32K the per-token payload is 10 KiB — payload
size is a non-issue for 10GbE.

## Measured cross-host round trip (labnode3 ↔ desktop, 2026-10-19)

`gate4_transfer_probe.py --host 127.0.0.1` on the desktop through an SSH
forward to labnode3 (firewalld on labnode3 admits 22 but not arbitrary high
ports; no passwordless sudo to open one — recorded, not worked around):

| payload | median RTT | max | min-bound effective rate |
|---|---|---|---|
| 10 KiB (decode token) | 42.1 ms | 83.5 ms | ~11 MB/s |
| 1.27 MiB (chunk 127) | 45.2 ms | 226.8 ms | ~31 MB/s |
| 1 MiB block | 44.1 ms | 46.0 ms | ~25 MB/s |

## Interpretation (honest)

- The ~42-45 ms floor is SSH-tunnel-mediated round-trip latency (two VLAN
  hops plus sshd userspace forwarding), NOT raw 10GbE throughput; the three
  payload sizes all sit at the same floor, so the measurement binds on RTT,
  not bandwidth.
- Gate-4 admission math at this RTT: a decode token needs TWO serialized
  boundary hops → ~84-90 ms/token floor through this path → ~11 tok/s
  ceiling — far below the qualified single-host A30+A2 pair and useless as a
  decode serving number. This does NOT condemn NCCL: NCCL over 10GbE uses
  its own socket/RDMA path with pipelined chunks, not an SSH userspace
  relay; the measured number is a routing-environment bound, not the
  transport's.
- Decision value: the gate-4 runner MUST qualify cross-host collectives over
  torchrun/NCCL directly (the #514/#515 rendezvous fixes), and the per-token
  transfer cost must be measured in THAT path before any throughput
  statement. Opening a firewalld port for direct NCCL is an owner action
  (passwordless sudo is unavailable to agents on labnode3).
- Payload sizes confirm the plan shape: halving the A2's stage (the #551
  balance) is the thermal relief; the transfer cost is latency-dominated, so
  chunked prefill (1.27 MiB per hop) amortizes far better than per-token
  hops — the runner should pipeline prefill chunks, not tokens.

## Reproduction

- Server: `python3 /tmp/gate4_transfer_probe.py --serve --port 9319` on labnode3.
- Client: `python3 /tmp/gate4_transfer_probe.py --host 127.0.0.1 --port 9320 --repeats 30`
  on the desktop via `ssh -N -L 9320:127.0.0.1:9319`.
- Raw report: `gate4-transfer-tunnel.json`.
