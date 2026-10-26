# MIG capacity cell — #609 gate 3 (plus gate-2 admission fix)

Date: 2026-10-25. Host: labnode3 A30 (harvested GA100, driver 595.58.03,
venv `/home/damen/dllm-dev/venv`, torch 2.13.0+cu130), repo tree staged at
`/tmp/mig609` from main `40f071f4` plus this PR's inventory/admission fixes.
The user granted the GPU window; MIG enable/disable was announced, scoped to
GPU 0 (A30) only, and reverted.

## What ran

1. **Inventory before any state change** (`post-state`-style pre-query in
   `cell-slice.out`): both GPUs idle, A30 MIG Disabled, A2 MIG N/A.
2. **MIG enable + carve**: `nvidia-smi mig -i 0 -cgi 14 -C` created ONE
   `MIG 1g.6gb` GPU instance (driver-reported geometry at inventory time:
   1g.6gb = 5.81 GiB / 14 SMs; 2g.12gb = 11.75 GiB / 28 SMs; 4g.24gb =
   23.56 GiB / 56 SMs — nothing hardcoded from marketing tables).
3. **Slice phase** (`mig-cell-slice.json`, runner `mig-cell-609.py`):
   - torch sees the slice truthfully: name `NVIDIA A30 MIG 1g.6gb`,
     14 SMs, `total_memory` 6,241,124,352 bytes.
   - The cotenancy inventory reports the A30 as MIG-capable per row with
     the live slice enumerated unprivileged (`nvidia-smi -L`).
   - Qualified fixture Qwen3.5-0.8B fp16 (rev `2fc06364…`), ctx 2048,
     batch 1, `--no-graph`: 3/3 fixed greedy prompts correct on the slice;
     slice-local accounting (1684 MiB on the slice; free 6.117 → 4.345 GB
     during the serve); clean release (no compute apps after terminate).
4. **Over-slice admission (request-driven — lazy residency means a bound
   port proves nothing)**: gpt-oss-20b MXFP4 (~12.6 GiB weights alone),
   ctx 512, `--host-gib 16` so the refusal is not the 8 GiB host-weight
   gate (that stock-budget refusal is recorded separately in
   `serve-slice-20b-hostgate.log` / `mig-cell-slice-hostgate.json`).
   Result on the slice: `MemoryError: insufficient GPU memory for
   weights, KV cache, and workspace reserve` from executor construction —
   the serve stayed alive; the REQUEST was refused.
5. **MIG restore**: GPU instances destroyed, MIG disabled, verified
   (`post-state.csv`, `post-devices.txt`): A30 mode Disabled, 0 MiB, full
   card visible again, no MIG devices, both GPUs idle, Default compute
   mode — the exact pre-cell state.
6. **Full-card control phase** (`mig-cell-full.json`):
   - Full A30: `NVIDIA A30`, `total_memory` 25,338,052,608 bytes (23.56 GiB).
   - The SAME 0.8B fixture/prompts: greedy tokens IDENTICAL to the slice
     run (same silicon, slice vs whole card).
   - The SAME 20B request that the slice refused: **ADMITTED** on the full
     card (executor constructs, generates). The refusal priced against
     the SLICE total, not the model or the request.

## Verdict — gate 3 passed; gate 2 fix shipped

A MIG slice is a real admission domain: the slice total (5.81 GiB) is
what admission prices against, greedy equivalence holds slice↔full, and
release is clean. Two shipped bugs the live cell caught (both fixed in
this PR):

- `compute_mode_inventory`/`mps_inventory` passed `--format=csv,noheader`
  to a parser that reads the first line as the header — every NVIDIA host
  KeyErroded in `compute_mode` (only hosts WITHOUT nvidia-smi, like HOMER,
  passed). Fixed to header format + regression test.
- `mig_inventory` treated ANY `N/A` in the whole query output as
  "unsupported": on a mixed host (A30 MIG-capable + A2 consumer) it
  reported MIG unsupported WHILE a live slice was serving. Fixed to
  per-row parsing with unprivileged `-L` enumeration + regression test.
- `require_gpu` admitted only `GPU-` UUID prefixes, so a `MIG-…` slice
  UUID could not even start a serve. Fixed: single-slice MIG admission
  allowed; multi-device (pipeline/distributed) still requires full
  `GPU-` UUIDs (slices have no P2P — the topology planner treats them as
  separate hosts). Tests added.

## Honest scope

- No throughput claim: MIG vs full-card latency/throughput was not
  measured (the #609 stopping condition is capacity/isolation behavior).
- torch's `get_device_properties(0).uuid` reports the PARENT GPU UUID on
  a MIG device; the truthful slice identity is the `MIG-…` UUID in
  `CUDA_VISIBLE_DEVICES` (recorded in the runner).
- The first two refusal probes (Qwen3.5-2B at ctx 8192/32768) only waited
  for the port — with lazy residency those were NOT admissions and are
  not claimed as findings; the recorded refusal is the request-driven
  20B one.
- Slice keep-out: the A30 had 36 MiB MIG-structural overhead while
  enabled; memory.used returned to 0 MiB after disable.
- A2 untouched throughout (MIG N/A; never addressed by any command).
