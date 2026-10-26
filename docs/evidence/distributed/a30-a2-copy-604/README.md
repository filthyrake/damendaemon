# A30→A2 boundary copy cell — #604 gate 4

Date: 2026-10-25. Host: labnode3 (harvested-GA100 A30 + harvested-GA107 A2
in one VM, driver 595.58.03, venv `/home/damen/dllm-dev/venv`, torch
2.13.0+cu130), repo tree staged at `/tmp/g604` from main `fedcf2a8`.
User-granted GPU window; read-only on device state (the only mutation is
the gate-3 policy's own disclosed peer-access ENABLE ATTEMPT, which the
runtime reports as unavailable before any enable is attempted).

Runner: `cell-604-gate4.py`; records: `cell-gate4.jsonl` (every event,
including the peer plan/enable disclosures). Medians over 21 measured
iterations after 5 warm, synchronized on both devices; payload integrity
asserted via host comparison after measurement. `if not __debug__` guard
per house convention.

## Finding 1 — P2P is unavailable on this pair

`can_device_access_peer` reports false for BOTH directions (A30→A2 and
A2→A30): the VM's passthrough exposes no peer access. This is the
recorded placement finding the issue anticipated. No driver error is
involved — the runtime simply reports the pairs as unreachable.

## Finding 2 — the driver-staged direct copy wins D2D; pinned staging does not

Boundary copy bandwidth (GiB/s, medians, warm):

| Payload | Direction | Blocking-unpinned (driver-staged) | Pinned serial two-hop | Pinned chunk-pipelined |
|---|---|---|---|---|
| 1 MiB | A30→A2 | 3.38 | **3.54** | 2.78 |
| 1 MiB | A2→A30 | 3.08 | **3.52** | 2.62 |
| 8 MiB | A30→A2 | **5.26** | 3.82 | 4.75 |
| 8 MiB | A2→A30 | **5.44** | 3.91 | 5.11 |
| 32 MiB | A30→A2 | **5.60** | 3.89 | 5.08 |
| 32 MiB | A2→A30 | **5.80** | 3.96 | 5.76 |

- At boundary sizes ≥ 8 MiB the DIRECT blocking-unpinned copy (the
  driver's own internal host staging) is the fastest path on this pair.
  The pinned two-hop loses ~30% serial; 1 MiB chunk pipelining recovers
  most of it (0.90–0.99× of blocking) but never beats the driver.
- At 1 MiB the serial pinned hop is the only variant that beats blocking
  (a small-payload latency effect, ~1.05–1.14×); pipelining hurts there
  (per-chunk launch overhead dominates).
- The issue's example shape — "pinned staged transfers become the default
  for that pair" if P2P is blocked — is REFUTED by measurement for D2D
  boundaries on this VM: the shipped gate-3 default (direct copy for D2D,
  pinned staging opt-in for HOST round-trips, where the desktop cell
  measured 2.84×) is the right placement and stays.

## Finding 3 — NUMA characterization: disclosed skip

The VM exposes a single NUMA domain (`/sys/devices/system/node` →
node 0 only): there is no NUMA-local vs NUMA-remote distinction to
characterize, recorded as a disclosed skip (the same honest treatment as
the NUMA-flat desktop in the gate-1/2 inventory).

## Verdict — gate 4 delivered; #604's stopping condition met

Measured A30→A2 copy cell with before/after paths, medians over warm
channels, reported separately from compute (pure copy walls; no model
was loaded), plus the NUMA skip disclosure. Combined with gates 1–3
(inventory + planner penalties, PR #627; transfer-path selection +
bounded pinned staging, PR #630), the issue's stopping condition —
"inventory + planner input + the measured copy cell" — is complete.
No throughput-parity claims follow.
