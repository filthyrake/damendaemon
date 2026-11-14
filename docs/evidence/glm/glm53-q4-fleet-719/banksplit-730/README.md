# #730: dual-3090 NVLink peer-resident q4 bank split, layers [58,66)

2026-11-14, 3090box (2× RTX 3090, NV4), guarded (`guard_shared_cuda`, both
GPU UUIDs explicit, 90 °C / reserve 0 / 1800 s). Runner + all attempt
reports retained here.

## Result: mechanism PROVEN bit-exact; memory headroom tighter than budget

The split arm made the whole range's q4 banks device-resident (40.66 GiB
aggregate, contiguous expert halves per GPU, 21.83 GiB each — infeasible on
one 24 GB card), executed on cuda:0, and read far-half experts on demand
over NVLink peer access.

**Gate 1 — the headline: bit-exact vs the shipped streamed control.**
`torch.equal(split_final, parked_3b_dump)` TRUE, max_abs 0.0 exactly
(rung-4a's streamed==resident class holds through the P2P topology).
Per-layer paired-fp8-oracle parity 8/8 green, worst rel RMS 0.1331 —
bit-identical to 3b's own recorded number for these layers: the
deterministic chain reproduces exactly through a different memory
placement. Peer integrity round-trips bit-identical (8/8 probed experts).
Aggregate peer bandwidth 15.69 GiB/s over 174 far reads (3.69 GB moved) —
NVLink engaged well above the 1.32 GiB/s host-staged class, below the
47.52 GiB/s 2-GiB-blob qualification as expected at per-expert (~32 MB)
transfer sizes.

**Gate 4 — failed as declared, and the finding is the deliverable.**
The runner's conservative line (peak reserved ≤ 94% of device memory)
was exceeded: peaks 24.89 GB (96.6%) / 24.70 GB (95.7%). The issue's
pre-declared acceptance bound — per-GPU peak within the GUARD envelope —
held: guard completed clean (633 s), no OOM, no reserve breach at
`--gpu-reserve-mib 0`. Both numbers are stated; the report's
`passed: false` is the strict in-runner line and is left unedited. The
measured headroom (~3.4%) is the real capacity bound: this 8-layer /
40.66 GiB range fits, a 9-layer range would not. #731's placement
arithmetic should consume the measured headroom, not the assumed 6%.

## Attempts (all retained)

| # | outcome | seconds | cause |
|---|---|---|---|
| 1 | refused in layer 58 | 31 | bank_get factory signature (hook passes 2 args) |
| 2 | refused after layer 58 | 100 | `flush=True` inside `json.dumps` — but proved the full first-layer compute + parity |
| 3 | all compute gates green | 633 | the record above |

Attempts 1–2 cost zero parity compute; both were print/plumbing-level,
found by init-verification within seconds of the first layer.

## Capacity record (gate 5)

- Aggregate resident banks: 43,656,413,184 B (40.66 GiB) across 8 layers
  (5.08 GiB/layer, 256 experts, contiguous split at expert 128).
- Per-GPU: 21.83 GiB banks + ~2.9 GB working set (fp8 oracle transients +
  allocator chunks) → peak 96.6%/95.7% of 24 GB.
- Single-GPU equivalent: impossible — 40.66 GiB > 24 GB; the shipped path
  streams from host RAM per use instead.

Parents: #730 (this cell), #719 (machinery + control), #599 (interconnect).
No throughput/SLO/quality claim — capability + parity only.
