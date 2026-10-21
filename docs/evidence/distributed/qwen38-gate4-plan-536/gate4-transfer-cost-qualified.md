# Gate 4 — per-token host-transfer cost decomposition (#536 cycle 9)

Date: 2026-10-20 (UTC)
Run: `scripts/experiments/gate4_transfer_cost.py` under torchrun on the
qualified three-rank chain (labnode3 A30 rank 0 coordinator + A2 rank 1,
desktop RTX 3090 rank 2 cross-host over 10GbE).

## Microbench — pure collective cost at the chain's payload shapes

Warmup 10 + 100 timed iterations per shape, synchronized CUDA streams:

| payload | origin rank 0 (intra-host) | origin rank 2 (cross-host) |
|---|---|---|
| decode hidden [1, 5120] fp32 (20 KiB) | 191.6 µs | 92.5 µs |
| prefill-chunk hidden [512, 5120] fp32 (10 MiB) | 35.7 ms | 35.6 ms |
| token row [1] int64 | 64.1 µs | 69.2 µs |
| MIN-ack all-reduce (int64) | — | 286.4 µs |

Readings:
- Small-payload broadcasts are **launch/sync-overhead dominated, not
  bandwidth-limited** — the cross-host origin is not slower than the
  intra-host origin at 20 KiB (algorithm/arrival variance exceeds the wire
  cost). A decode step's pure collective cost (2 hidden + 1 token + 1
  MIN-ack) is ≈ **0.75–1.1 ms**.
- The 10 MiB prefill-chunk hop costs ≈ 35.6–35.7 ms regardless of origin —
  the cross-host 10GbE link carries the ring; ≈ 290 MB/s effective for a
  3-rank NCCL broadcast, the wire (not launch) bound regime. Chunked
  prefill pays this per chunk boundary, not per token.

## Real decode — where the step wall time goes (instrumented, rank 0)

27B Q4, context 2048, batch 1, standard protocol; steady step
**137.55 ms** (first 4 steps excluded; Triton-compile first step 337 ms).
Per-exchange wall times on the coordinator, decode rounds only:

| exchange (per step) | mean | min | max |
|---|---|---|---|
| hidden ×2 | 45.9 ms | 0.38 ms | 132.1 ms |
| tokens ×1 | 19.0 ms | 17.1 ms | 40.7 ms |
| MIN-ack finish ×1 | 0.94 ms | 0.74 ms | 1.19 ms |

A chain broadcast is a synchronization point, so an exchange call's wall
time **includes waiting for the source rank's compute** — it is an upper
bound on transfer, not pure transfer. The profile decomposes the step:

- hidden src=0 (coordinator's own broadcast): ≈ 0.4 ms — near-pure
  collective cost, matching the microbench.
- hidden src=1: ≈ 91 ms (max 132) — **rank 1's A2 stage compute**, the
  bottleneck of the chain.
- tokens src=2: ≈ 19 ms — the 3090 head's compute (20 layers + output
  head); the RTX 3090 is ~5× the A2's throughput on this stage shape.
- MIN ack: ≈ 0.9 ms.

## Conclusion (acceptance gate 4)

**Cross-host transfer is not the chain's bottleneck.** Per decode token,
pure collective cost is ≈ 0.75–1.1 ms — under 1% of the 137.55 ms steady
step. The chain is compute-bound on the slowest per-device stage (the
A2's 20 layers ≈ 91 ms; the 3090's 20 layers + head ≈ 19 ms). Reported
separately from single-host numbers: the two-rank in-process A30+A2 pair
decodes the same path at ≈ 144–150 ms/step — the three-device split
(137.55 ms) is not slower than the in-process pair despite two collective
hops crossing a host boundary per step. Chunked prefill is the only
wire-bound regime (≈ 35.6 ms per 10 MiB chunk boundary hop).

No SLO claim; wall-clock characterization of this topology.
Evidence: `gate4-transfer-cost.json`.
