# #70 A30 Q4 32K C1 capacity gate — attempted, definitive OOM failure (2026-10-17)

## Verdict

The bounded 32K C1 capacity cell **does not fit on a single A30** at the pinned
configuration. The measured request failed with CUDA OOM mid-chunked-prefill.
Production context stays 512; no promotion. The 4K replay result (84/84,
`a310f0e`) is unchanged and unaffected.

## Pinned configuration

Pinned Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Q4
compensated (isolated `hybrid_q4_context_probe` serve path, production
admission untouched), context 32768, input 32,256 raw token ids
(sha256 `31836e963e26cd64…` of the id array, engine tokenizer, no truncation),
output 512, C1 (batch 1), prefill chunk 127, no graphs, single A30
`GPU-70f6a991-a4d7-9984-c2a1-6c474445d53a` under `guard_shared_cuda`. Preflight
passed: A30 headroom ~23.9 GiB (≥22 GiB floor), host MemAvailable ~106 GiB
(≥100 GiB floor). All five attempts used identical workload and settings.

## Definitive failure (attempt 5)

- Warmup request (512 ids, max_tokens 1) completed: 616.5 s total — activation
  is the dominant fixed cost, consistent with #497's measured ~591 s.
- Allocator after warmup: 21.71 GB current / 21.81 GB peak.
- The 32K request then prefilled for 120.7 s and failed:
  `torch.OutOfMemoryError: Tried to allocate 260.00 MiB. GPU 0 has a total
  capacity of 23.60 GiB of which 259.44 MiB is free … 21.97 GiB is allocated
  by PyTorch` — raised in `scheduler._chunked_group` →
  `HybridExecutor.prefill_chunk` → `forward`, i.e. during the chunked prefill
  itself, with 16 full-attention layers scoring against a growing KV.
- Post-failure allocator peak: 23.85 GB (22.2 GiB). The cell is ~260 MiB short
  of fitting; the failure is genuine capacity exhaustion, not a guard kill or
  a transient.
- Guard: phase `completed`, 755.1 s (`gpu_reserve_mib: 0` recorded).

## Attempt history (all retained; harness fixes only — no workload setting changed)

- attempt1: runner bug — prompt builder under-produced tokens (BPE merges
  across repetition boundaries); no GPU work.
- attempt2: runner client socket timeout (630 s) clipped the warmup during
  activation; the 32K request never started.
- attempt3: guard steady-state reserve (2560 MiB) killed the run at 655.9 s
  during activation. Led to the `--gpu-reserve-mib` guard flag (default
  unchanged; lower only for near-capacity experiments on hosts with no
  incumbent workload).
- attempt4: reserve 1024 MiB — killed at 656.3 s, last observed free 290 MiB,
  during the 32K prefill's memory ramp (~23 s after the request started).
- attempt5: reserve 0 (memory floor disabled; thermal/identity/host gates
  active) — the workload ran to its own definitive OOM, recorded above.

## Consequences

- The A30 32K C1 cell is closed as a measured capacity failure. Per the
  contract, no unchanged retry and no setting sweep (including
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, which the OOM message
  suggests — the 282 MiB reserved-but-unallocated vs 260 MiB ask makes it
  marginal, but a retry with changed settings is out of contract).
- Remaining paths to 32K for this model live under #60: sharding/offload
  (#51/#64 two-device path, now NCCL-qualified), HOMER cells under #68, or a
  separately contracted smaller pinned context cell.
- Production context remains 512.

## Files

- `q32k-gate5.json` / `q32k-guard.json` / `q32k-gate5.serve.log` / `q32k-runner.log` — attempt 5 (definitive OOM)
- `attempt1…4-*` — retained prior attempts
- All timings are wall-clock controls on one host state; capacity only, not
  quality or SLO evidence.
