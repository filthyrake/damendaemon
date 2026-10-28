# RDMA transport qualification: inventory state + probe/verification design (#602)

Status: **inventory-blocked, machinery validated**. The issue's acceptance
gate requires a real RDMA NIC pair; the fleet has none (verified live,
2026-10-27 — record below). Per the issue's own stopping condition, this
directory holds the inventory record, the probe/verification design, the
RoCE first-run checklist (`docs/rdma-roce-first-run.md`), and — new this
pass — the probe machinery validated end-to-end on real cross-host
hardware over the SOCKET transport (the qualified default), including
resolution of both blockers from the 2026-10-24 WIP incident.

## Fleet RDMA inventory (2026-10-27, live probes)

| host | GPUs | network hardware | `/sys/class/infiniband` | verdict |
|---|---|---|---|---|
| rocky10-desktop | RTX 3090 (+B580 display) | 2× Aquantia AQC113C 10GbE, WiFi 7 | empty | no RDMA endpoint |
| labnode3 (VM) | A30 + A2 | virtio NIC only | empty | no RDMA endpoint |
| HOMER | 8060S iGPU | RTL8125 2.5GbE, WiFi | empty | no RDMA endpoint |

`rdma_cm`/`ib_core` modules are loaded on two hosts by CIFS/Samba
consumers (cifs, smbdirect) — no RDMA device exists behind them. The
A2+3090 unlock (owner, 2026-10-27) makes both GPU hosts available for the
SOCKET cross-host cell but cannot create an RDMA endpoint; #602's RDMA
gates stay inventory-blocked, exactly as the issue body pre-declares.

## Cross-host socket cell: PASSED (2026-10-27) — the machinery is real

`scripts/probe_distributed_gpu.py`, rank 0 = labnode3 **A2**
(`GPU-1b29c7c8`), rank 1 = rocky10-desktop **3090** (UUID-pinned),
`NCCL_NET=Socket` forced, torch 2.13.0+cu130 / NCCL 2.29.7 both sides
(version-matched per the standing gate). Both reports
`passed: true`, 8/8 measurement cells, tensor assertions green, zero
NCCL WARNs either side. Artifacts in `socket-cell-2026-10-27/`
(`rank0.json` + transcript from labnode3, `desktop-rank1.json` +
transcript from the desktop, `bind2.log` bind-family watcher):

| operation | 4 KiB | 64 KiB | 1 MiB | 4 MiB |
|---|---|---|---|---|
| all_reduce (median) | 0.412 ms | 0.864 ms | 4.092 ms | 15.003 ms |
| round_trip (median) | 0.663 ms | 1.176 ms | 8.216 ms | 29.68 ms |

(4 MiB round trip ≈ 7 Gbit/s effective — a healthy 10GbE-attached number;
medians over 20 timed iterations after 3 warmups, per cell.)

### Parser reclassification note (honest-record rule)

The reports' own `transport_evidence` says `ib` — computed at run time by
the parser as deployed then, which over-matched NCCL 2.29.7's
`NET/IB : Using [RO]` **fallback claim** (printed even beside
`NET/IB : No device found.` on this RDMA-less host, while every data
channel ran `via NET/Socket/0`). The parser now anchors classification on
the channel lines (plugin `Using` lines are fallback-only); re-parsing the
retained transcripts (sha256-pinned in both reports, unchanged) yields
`socket` / `not_applicable` — the true verdicts. The recorded JSONs are
NOT edited; this note is the reclassification record.

## The 2026-10-24 incident: both blockers resolved, root causes measured

The WIP incident (`INCIDENT-2026-10-24-wip.md`) left two unresolved
blockers. Both are now closed with root causes, all found on the A2+3090
pair:

### 1. "Cross-host round_trip deadlock" → an IPv6 link-local proxy failure, not a deadlock

NCCL on the desktop tried to connect rank 0's proxy at
`fe80::2c50:77ff:fe07:d95a` (an IPv6 link-local from one of labnode3's
interface list — the host runs docker bridges too) and failed
`Network is unreachable` (the desktop has no IPv6 route), without
falling back to IPv4. The P2P `work.wait()` then spun — the "77-minute
deadlock" was this unreachable proxy connect. **Fix: `NCCL_SOCKET_IFNAME`
pinned to the real NIC on each rank** (labnode3 `enp6s18`, desktop
`enp72s0`) — with the interface restricted, the proxy addresses are the
real NIC's and the full grid completes with zero WARNs. Same
v6-preference family as the TCPStore bind below; recorded in the RoCE
checklist as a mandatory first-run item.

### 2. "0-byte transcripts on Linux" → NCCL logs to STDOUT, and the tee design itself was fragile

Measured: NCCL 2.29.7's C library logs to **fd 1** (stdout), not fd 2 —
the fd-2-only capture never saw a single NCCL line (reports read
`unknown` transport with the grid green). `StderrTranscript` is now
`ConsoleTranscript`: a DIRECT dup2 redirect of BOTH fd 1 and fd 2 into the
transcript — no pump thread (the pump/fork/NCCL-proxy interaction remains
the recorded suspect for the original 0-byte failure; a kernel-level
redirect has no userspace moving parts to break). The final cell's
transcripts hold 75/62 real NCCL lines and re-classify correctly.

### 3. New finding: this torch build's TCPStore binds IPv6-only — v4 rendezvous needs a pre-bound fd

`dist.init_process_group(init_method='tcp://10.30.100.3:29500')` on torch
2.13.0+cu130 binds the store an **IPv6-only wildcard** (`ss -tln6`
listening, `ss -tln4` empty — measured for BOTH the libuv and classic
daemons, and `?use_libuv=0` does not change it): IPv4 peers get instant
ECONNREFUSED while rank 0's in-process client never needs the socket —
the store then times out "1/2 clients joined". A plain python listener on
the same port connects fine from the desktop, isolating the bind. The
probe now pre-binds an **AF_INET listener on rank 0 and hands c10d the fd
via `master_listen_fd`** (supported in the TCPStore constructor), plus a
`--store-timeout` client retry window that covers the peer's
torch-import + CUDA-init lead time (which also retires the incident's
fragile "poll for the listener" orchestration).

### 4. The deadline watchdog: live-fired once, and it worked — with one recorded miss

The first fixed-rendezvous attempt still stalled at the fe80 proxy
connect — and the new `DeadlineWatchdog` fired exactly as designed at
120 s on rank 0: labeled marker + full thread tracebacks into the transcript
(`rank0.json.nccl-debug.log` from that attempt, retained in the staging
log below), process exit 70, no 77-minute silent hang. The measured
hang point (barrier inside the first all_reduce cell) is what pointed at
the proxy path. Every guarded region in the probe is deadline-wrapped
now; `transcript_sha256` pins every report to its transcript.

One miss, recorded: rank 1 of that same failed attempt did NOT exit —
its process was found still alive hours later (holding 409 MiB on the
3090) and was killed manually (pid 1147281, 2026-10-27). The surviving
rank's stall outlived its own cell watchdog; watchdog coverage on the
peer that loses its partner first is a known open gap, not a solved
problem — the outer process deadline (`timeout`/job runner) stays
mandatory for unattended cross-host runs.

Retained raw logs of the debugging trail (labnode3 `/tmp/rdma602-v2/`,
not all copied): `bind2.log` (the ss-family watcher that caught the
v6-only bind), the failed-attempt console logs with the fe80 WARNs, and
the watchdog-fire transcript.

## What remains hardware-gated (unchanged)

No RDMA qualification is claimed from this socket cell — the issue's
stopping condition holds. When a ConnectX-class pair lands in the
desktop + labnode3: run the probe `--transport rdma` both ways with
transcripts retained (the machinery is now proven to capture and classify
them), then the boundary-burst / chunked-prefill 8K cell on the
cross-host chain, reported separately from compute (gate-4 decomposition
format). The RoCE first-run checklist (`docs/rdma-roce-first-run.md`)
now carries the three live-earned items: `NCCL_SOCKET_IFNAME` pinning
(the fe80 trap), the master_listen_fd v4 rendezvous, and the
`NCCL_SOCKET_IFNAME`+IFNAME combination for proxy paths.
