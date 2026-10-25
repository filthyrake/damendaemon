# #604 gates 1-2: read-only topology inventory + topology-aware planning (2026-10-24)

## Delivered

1. **Gate 1 — inventory**: `damendaemon/topology_inventory.py`, composed into
   `damendaemon preflight`. Per device: name/UUID, PCI bus id (normalized to
   the sysfs 4-digit domain form), NUMA node + CPU affinity from sysfs, the
   peer-access matrix (`can_device_access_peer`, per-pair errors recorded
   honestly), NVLink presence, PCIe link width/generation, virtualization
   markers (ACS/IOMMU P2P caveat), and NIC NUMA placement. Everything is
   read-only; readers are injectable so CI runs the full report with no GPU
   and no sysfs.
2. **Gate 2 — planner input**: `choose_stage_splits` gains opt-in
   `boundary_topology` (+ `topology_policy`): non-P2P links carry a 1/8
   penalty and cross-NUMA links 1/16, disclosed in the plan as
   `topology_penalty_total` and per-boundary `boundary_links`;
   `topology_policy='exclude'` refuses orderings whose adjacent pair lacks
   P2P. The default call path is the legacy DP byte-for-byte.

## Live verification (desktop 3090, driver 615.71.09)

`damendaemon preflight` on the desktop reported, from real sysfs/nvidia-smi:
device RTX 3090 at `0000:81:00.0`, cpus 0-63, **NUMA-flat machine (sysfs
node -1)** — disclosed as "no cross-socket staging cost" rather than an
unknown; peer matrix empty (single device, the honest case); no NVLink;
PCIe x16 Gen4; not virtualized; NICs listed with node -1.

Debugging notes along the way: nvidia-smi pads the PCI domain to 8 digits
(sysfs uses 4) and the address itself is colon-separated, so naive
normalization produced `0000:00.0`; sysfs `-1` means NUMA-flat, not missing.

## Tests

44 CPU tests green: 12 topology (inventory + planning), 6 co-tenancy
(#609 compose updated for the appended item), existing placement and
rank-precision suites unchanged.

## Honest boundaries

- Gate 3 (transfer-path enablement + pinned staging in the runtime) and
  gate 4 (the measured A30→A2 boundary cell, NUMA-local vs NUMA-remote host
  staging on labnode3) are **not** in this PR: gate 4 needs the
  user-reserved A30/A2 pair. The inventory is the input both consume when
  that window opens.
- No multi-GPU P2P measurement exists yet on any host: the desktop has one
  CUDA device, so the peer matrix is exercised by unit tests only.
