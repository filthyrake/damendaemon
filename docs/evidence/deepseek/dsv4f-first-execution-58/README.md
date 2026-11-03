# #58 — First execution evidence: teacher-forced forward through real official weights, all gates pass

**Fourth physical-qualification increment** (after #702 contract audit, #703
fleet plan, #704 real-payload prepare): the first #58 cell to EXECUTE the
reference forward against the official `DeepSeek-V4-Flash-0731` weights —
32 teacher-forced tokens through layer 0 (ratio-0 sliding-window) and layer
2 (the first ratio-4 compressed-attention layer), via the production
`DeepSeekStageSlots` serving-shaped path.

## Gates (pre-declared) — ALL PASS

1. **Execution**: 32-token prefill completes through both stages; outputs
   finite; shapes exact `(1, 32, 4, 4096)` (4 hyper-connection streams ×
   4096 dim). Layer 0: 333.35 s; layer 2: 328.30 s (fp32 tile-wise CPU
   reference, 8 threads — the reference executor, not a serving kernel;
   ~10 s/token/layer is the documented reference cost, not a perf claim).
2. **Compression-state parity on real weights**: `prefill(31) + decode(1)`
   last-position hidden vs full `prefill(32)` last-position hidden —
   - layer 0 (ratio-0): max |Δ| 2.98e-7 (2.13e-6 relative, |h| mean 0.140)
   - layer 2 (ratio-4): max |Δ| 3.87e-7 (2.67e-6 relative, |h| mean 0.145)
   The decode path reproduces the prefill path through the compressed-cache
   machinery at real geometry — fp32-noise-level agreement, both boundary
   kinds (window and compression).
3. **Reload determinism**: full stage reload from the NAS reproduces both
   stages' 32-token prefill outputs **byte-identically**.

## Setup

- Host: labnode3 (CPU-only; the A30 lane untouched), `/home/damen/dsv4f-venv`
  (python3.11, CPU torch 2.15), staged tree `/tmp/damendaemon-58` at
  `389efe0f`, read-only NAS source.
- Tokens: deterministic synthetic IDs `(i*7+3) % 129280` — real vocab space,
  no natural-language claim; tokenization quality is a separate surface.
- Stages: `load_reference_stage(0,1)` (embedding boundary) and `(2,3)`
  (mid), driven through `DeepSeekStageSlots.forward_layers` (the
  serving-shaped slots path: admission, cache reservation, prefill/decode
  branches) — 9.0 GiB residency, 17.7 GiB peak, released on cleanup.

## Honest scope

- Two layers of a 43-layer model, teacher-forced, hidden-stream parity only —
  no logits (mid-model), no generation, no quality verdict, no throughput
  claim (reference executor). Full-model logits, the two-process host
  topology, and quality tiers remain open.

## Artifacts

- `cell-c.log` — full gate output incl. per-gate numbers
- `dsv4f_exec_cell_c.py` — the cell (gates pre-declared in the docstring)
