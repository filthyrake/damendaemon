# Dual RTX 3090 NVLink interconnect qualification (#599)

2026-11-14, 3090box (2× RTX 3090, NV4 NVLink bridges), guarded
(`guard_shared_cuda` with both explicit GPU UUIDs, thermal 90 / reserve 0).
5.2 s, peak reserved 2 GiB.

## Result — ALL GATES PASS

1. **Topology: NV4** — nvidia-smi topo reports NVLink between GPU0↔GPU1.
2. **Peer access**: `can_device_access_peer` true both directions. (torch
   2.10 has no `enable_peer_access` — peer access for cross-device copies
   is driver-managed; the measured copy is the observable.)
3. **Bandwidth: 47.52 GiB/s** device-to-device (2 GiB payload) vs
   **1.32 GiB/s** host-staged (pinned CPU) control — **36.1×**, proving
   NVLink is genuinely engaged, not host-routed.
4. **Integrity**: peer-copied payload bit-identical.

Attempt 1 (retained in #599 comment): refused at 3.3 s — my topology
parser read the self-column instead of the interconnect column, and I
called a torch API (`enable_peer_access`) that doesn't exist in 2.10.
Both fixed; raw probe numbers matched the guarded rerun.

## Significance for #599

The fleet's first inter-GPU interconnect is qualified — #599's "No
interconnect topology anywhere in placement" gap now has a measured,
guarded foundation to build on. Practical consequences:

- 3090box presents 48 GiB aggregate VRAM with 47.5 GiB/s between halves —
  a dual-GPU expert-bank split for #719 stage 3 (or whole-model cells
  up to ~48 GiB of resident banks) is now a qualified design option.
- Host-staged fallback measured at 1.32 GiB/s — 36× slower — justifies
  peer-aware placement policy over naive staging.

No serving/SLO/placement-policy claim — capability qualification only.
Follow-ups (separate issues when wanted): guarded dual-GPU q4 bank-split
cell for #719 stage 3; interconnect-aware placement in the #599 topology
layer (`can_device_access_peer` is now a qualified observable).
