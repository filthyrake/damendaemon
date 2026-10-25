# #609 MPS decision cell: qualified on the desktop 3090 (2026-10-24)

## Verdict

**The MPS half of #609 is QUALIFIED, with measured numbers:** two concurrent
damendaemon serve processes ran correctly through the MPS server, per-process
memory attribution stayed bounded and identical to the baseline, and the
concurrent-generation contention delta was a small but real win for MPS.
No claim of throughput improvement beyond this bounded cell.

## Cell (desktop 3090, driver 615.71.09, Qwen2.5-0.5B, --no-graph)

1. **Inventory (gate 1, live via `damendaemon preflight`)**: MIG not
   supported on the consumer card (honest N/A — slice admission stays an
   A30/data-center path); compute mode Default; MPS control binary present,
   no server running. Read-only, no state change.
2. **Baseline (no MPS)**: two serves on one GPU, both correct (identical
   greedy output), 1386 MiB per process, 2800 MiB total.
3. **MPS**: both serves correct through `nvidia-cuda-mps-server`
   (server process: 30 MiB); per-process attribution 1386 MiB each — on
   driver 615.71 nvidia-smi attributes memory per-client, so **no
   measurable context-memory saving is visible at the query level** for a
   0.5B model.
4. **Contention probe** (two concurrent 200-token generations, wall time):

| mode | client walls | total wall |
|---|---|---|
| baseline (time-slicing) | 2.22 / 2.22 | 2.22 s |
| MPS | 1.99 / 2.01 | **2.01 s (~10% faster)** |

## Decision

MPS is admitted for co-tenant damendaemon-family serving on this host:
correct output, bounded memory, kernels overlap instead of time-slicing.
The win at 0.5B scale is modest; the primary MPS benefit for larger models
(context sharing) is not visible through nvidia-smi attribution on this
driver and would need an in-process accounting check to claim.

## Honest boundaries

- The cell served with `--no-graph`; the CUDA-graphs × MPS interaction is
  untested and remains the known risk area — do not assume graph capture
  works under MPS.
- MIG half of #609 (inventory on the A30, slice-aware admission, guarded
  slice cell) was not started: the A30/A2 are user-reserved. The
  inventory code ships here and runs read-only when that window opens.
- No throughput claim beyond the contention numbers above; capacity/
  isolation behavior is the recorded result, per the issue's stopping
  condition.

Files: `mps-cell-{baseline,mps}.log`, `contention-{baseline,mps}.json`
(on the host at /tmp/mps609/ and mirrored below where fetched).
