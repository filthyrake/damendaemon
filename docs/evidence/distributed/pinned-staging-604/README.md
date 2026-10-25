# #604 gate 3: transfer-path selection + bounded pinned staging (2026-10-25)

## Delivered

1. **Peer access enablement**: `transfer_policy.enable_peer_access` enables
   P2P for every mutually-capable ordered pair via
   `cudaDeviceEnablePeerAccess` (error 701/already-enabled tolerated, every
   other outcome disclosed, never raised) and returns per-pair disclosures.
   Wired into `HybridPipeline` construction — cross-stage `.to()` transfers
   take the P2P path where the fleet geometry supports it, fall back to the
   staged driver path otherwise. The disclosure lives on
   `pipeline.peer_disclosures`.
2. **Bounded pinned staging** (opt-in `DAMENDAEMON_PINNED_STAGING=1`, strict
   tri-state, default off): `PinnedStaging` stages the `cuda_host_stage`
   host-boundary round-trips through pinned host buffers bounded by the
   caller's existing `max_transfer_bytes` budget — one buffer per direction,
   grown lazily, never beyond the budget. D2H clones out of the pinned
   buffer so callers still receive ordinary unpinned contiguous CPU tensors,
   matching the boundary contract. Disabled keeps the historical pageable
   copy byte-for-byte. Allocation and H2D are module-level seams so CPU-only
   contract runs mock them.

## Live cell (desktop 3090, real cudaHostAlloc, bounded: 8 MiB × 32 × 3 rounds)

| mode | round-trip bandwidth | speedup |
|---|---|---|
| pageable (historical) | 5.16 GiB/s | 1.00× |
| **pinned staging** | **14.65 GiB/s** | **2.84×** |

Serve smoke with `DAMENDAEMON_PINNED_STAGING=1`: correct greedy output,
normal latency; numerical identity asserted across every round-trip chunk
in the bandwidth cell (`torch.equal` per chunk).

## Honest boundaries

- Peer enablement could not be exercised live on a multi-GPU pair: the
  desktop has one CUDA device (disclosure list empty, the honest single-GPU
  case) and the A30/A2 pair is user-reserved. The enablement path is
  covered by mocked-contract tests (mutual enable, 701 tolerance, error
  disclosure, fallback).
- Pinned staging is opt-in and NOT yet enabled by default in production
  serving; flipping the default is a follow-up once a chain cell exercises
  it end-to-end (needs the reserved pair).
- The bandwidth cell measures the transfer path in isolation — it is not an
  end-to-end serving claim.

14 CPU tests (policy tri-state, budget bounding, round-trip identity,
peer-plan honesty, enablement disclosure); host-stage/pipeline suites
unchanged and green.
