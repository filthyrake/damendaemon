# First-run checklist: homelab RDMA (RoCEv2) for cross-host ranks

Status: **inventory-blocked** (#602). No fleet host currently exposes an
RDMA device (`/sys/class/infiniband` empty on both labnode3 and
rocky10-desktop; no ConnectX-class NIC in lspci). This checklist is the
design work the issue holds until a real RDMA pair exists — it is not a
qualification record and must never be cited as one. Socket transport
remains the qualified default; RDMA is opt-in only
(`probe_distributed_gpu.py --transport rdma`), and the probe refuses RDMA
on a host without an RDMA device.

## 1. Link layer (before anything RDMA)

- [ ] Two ConnectX-class NICs, one per host, direct-attached or via a
  switch that carries the full MTU (no IB switch is assumed: RoCEv2 over
  Ethernet is the homelab endpoint).
- [ ] Jumbo frames end-to-end: set MTU 9000 on both NICs **and** every
  switch hop in between; verify with `ping -M do -s 8972 <peer>`. A single
  1500-MTU hop silently degrades RoCE to fragmentation.
- [ ] Driver loaded and ports up: `ip link set <dev> up`,
  `ibdev2netdev` shows the mapping; `rdma link` reports the device.
- [ ] If a switch is in path, PFC or a lossless policy for the RoCE
  traffic class; on direct-attach, PFC is usually unnecessary.

## 2. RoCEv2 specifics

- [ ] GID index: with `ibv_query_gid`/`show_gids`, find the RoCEv2 (UDP
  port 4791) GID bound to the routable VLAN/IP; export
  `NCCL_IB_GID_INDEX=<idx>` per host if auto-selection picks the wrong one.
- [ ] Traffic class / DSCP: `NCCL_IB_TC=106` (0x6A, the common RoCE
  lossless class) matching whatever the switch PFC profile expects.
- [ ] HCA selection when multiple ports: `NCCL_IB_HCA=<name>` to pin the
  intended port; otherwise NCCL probes by bandwidth.

## 3. NCCL environment

- `NCCL_NET=IB` (what the probe's `--transport rdma` sets) — or
  `NCCL_NET=Socket` to force the socket control path for A/B comparison.
- `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET` (the probe sets both);
  the probe captures fd 2 per rank to `<output>.nccl-debug.log` and
  classifies it into the report (`transport_evidence`).
- Keep the SAME torch/NCCL build on every rank — cross-version groups fail
  with opaque `ncclSystemError` socket errors, not a version diagnostic.
- NCCL listeners bind ephemeral ports; firewall rules must cover the
  rendezvous port AND the ephemeral range scoped to the peer's address
  (labnode3 rich-rule pattern), or a custom
  `NCCL_PORT_RANGE=<lo>-<hi>` with matching rules.

## 4. Verification (what proves what)

- Transport actually selected: transcript line `NET/IB : Using [N]<hca>:<port>/RoCE v2 ...`
  or per-channel `via NET/IB/<n>`; the probe records these in
  `transport_evidence.ib_selection_lines` and sets
  `nccl_selected_transport: "ib"`.
- GPUDirect capability, NCCL's own check: a `GPU Direct RDMA enabled for
  HCA <n> '<hca>'` line → the probe records `gpudirect_rdma: "evidenced"`.
  Absence of the line with IB selected → `"unknown"`; NCCL ran its check
  and refused → `"not_available"`. Per-HCA verdict lines are all retained.
- What this does NOT prove: per-transfer DMA-bypass of the host
  translation path. For that, compare the probe's round-trip medians
  socket-vs-RDMA for the 1–4 MiB sizes and look for the GPUDirect delta on
  top of the bandwidth delta; an RDMA-selected run alone is not a GPUDirect
  claim (NCCL itself warns via its own probe wording).
- Qualification shape when hardware lands: probe socket-forced AND
  rdma-forced, both directions, transcripts retained in
  `docs/evidence/distributed/`; then the boundary-burst/chunked-prefill
  cell at 8K on the cross-host chain reported separately from compute
  (gate-4 transfer-decomposition format).

## 5. Known fleet constraints

- Desktop 3090: FP unlock active — treat any transport benchmark on it as
  its own hardware epoch; do not mix pre/post-unlock numbers.
- Gate-4 finding: decode on the A2 stage is compute-bound (collective
  ≈0.75–1.1 ms/token, <1% of the 137.55 ms step) — RDMA's win case is
  prefill ramps and future KV/prefix handoff, not decode.
- The probe's inventory gate refuses `--transport rdma` on any host whose
  `/sys/class/infiniband` is empty — keep that refusal intact; it is the
  honest no-simulation boundary.
