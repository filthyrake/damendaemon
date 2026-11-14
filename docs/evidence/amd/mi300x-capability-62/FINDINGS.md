# MI300X capability evidence (issues #62, #719)

First real-hardware evidence from the MI300-class datacenter tier. Collected
2026-11-14 on host `amd-mi300x` (shared production box, 7 healthy GPUs, vLLM
serving stack on ports 8400-8411). Run via user-executed bounded probes
(`scripts/mi300_collect/`); the approved capability run executed inside the
production CRI container as UID 2000 with one visible GPU (`ROCR_VISIBLE_DEVICES=0`),
a 1 GiB PyTorch allocator cap, two CPU threads, `nice 19`, private caches and
a 180 s timeout. Full provenance: `STATUS.local.md`, `prepared-command.txt`,
`prepared-capability-run.py`; full bundle in `mi300-evidence-complete.tar.gz`
(SHA-verified against the external manifest `SHA256SUMS.files.txt`).

## Host and stack (pinned, #62 item 1)

- 8x AMD Instinct MI300X slots; **7 healthy** (`gfx942`, `sramecc+:xnack-`,
  304 CUs, 192 GiB HBM3 each = ~1.31 TiB total fleet VRAM). One device
  (PCI 0000:9d:00.0) is quarantined/absent.
- Full **xGMI mesh** interconnect, all-pairs 1 hop (rocm-smi topo) - the
  #62 "local multi-GPU collectives" tier has the hardware fabric for it.
- Host ROCm **7.2.1**; container runtime: PyTorch 2.12.0+git6bbd260,
  **HIP 7.2.53211**, **Triton 3.7.1**. hipBLASLt, RCCL, MIOpen present in
  /opt/rocm(-7.2.1)/lib.
- 2x Xeon Platinum 8480+ (56C/112T each), 2 NUMA nodes, 2 TiB host RAM
  (~1.8 TiB available at collection time).
- Serving co-tenancy reality: vLLM stage-engine + TP-worker processes hold
  ~180-188 GiB VRAM on 6 of 7 GPUs; probe ran in an idle window on GPU 0
  (15.61 GiB free at probe start, 1 GiB allocator cap respected).

## Measured results (probe_results.json, `probes-approved/`)

- **dtype scan**: all 9 dtypes cast/roundtrip on GPU, including all four
  FP8 variants (e4m3fn, e5m2, e4m3fnuz, e5m2fnuz).
- **FP8 semantics, measured on-device** (the #719 CUDA->ROCm boundary
  discriminator): `e4m3fn` 1.0 = `0x38`, max 448, NaN bytes {0x7f, 0xff};
  `e4m3fnuz` 1.0 = `0x40`, max 240, NaN byte {0x80} only. Matches CPU-torch
  reference tables - the fn/fnuz boundary must be handled explicitly in any
  cross-vendor FP8 tensor handoff; fnuz is the native AMD variant.
- **Triton q4 dequant** (damendaemon q4_0 shape: fp16 scale + packed
  nibbles): compiled on the AMD backend, **max abs err 0.0** vs plain-torch
  reference. Compiled artifacts captured at all four stages
  (TTIR/TTGIR/AMDGCN/HSACO). gfx942 codegen uses SDWA-optimized scalar
  nibble ops, 7 VGPRs/14 SGPRs - the #719 "dequant kernels on Triton
  paths" premise holds on this hardware.
- **bf16 matmul anchor**: max abs err 0.24 vs fp32 on a 256^3 case (normal
  bf16 rounding magnitude for this range).
- **FP8 `torch._scaled_mm`: RESOLVED by run 2 (probe v2, same day 2026-11-14).**
  Run 1's 32/32 rejection was a probe defect (mixed scaling shapes); run 2
  with corrected tensorwise/rowwise shapes measured the real support
  matrix on 64 cells (16 dtype pairs x 2 layouts x 2 scaling modes,
  256^3, bf16 out): **FP8 matmul works on this stack, fnuz-only,
  column-major-B.** Five cells executed: `e4m3fnuz x e4m3fnuz` (col-major,
  tensorwise AND rowwise), `e4m3fnuz x e5m2fnuz` (col, tw),
  `e5m2fnuz x e4m3fnuz` (col, tw), `e5m2fnuz x e5m2fnuz` (col, tw).
  Rejections: 20 row-major B ("only row-major x column-major", the
  cuBLASLt layout contract), 24 B-dtype ("Expected b e4m3fnuz" — the
  CUDA-native `fn` variants are rejected as B under every combination;
  ROCm hipBLASLt hard-requires the AMD fnuz encoding), 14
  HIPBLAS_STATUS_NOT_SUPPORTED (remaining non-fnuz A combos), 1 no
  hipblaslt solution (`e5m2fnuz x e4m3fnuz` col+rowwise). Rowwise scaling
  is supported only for the pure `e4m3fnuz x e4m3fnuz` pair. Measured max
  abs err vs fp32 on unit scales (2.9-6.2) includes FP8 quantization noise;
  no accuracy threshold is applied and no throughput is claimed. Benches
  still NOT RUN - D2D bandwidth and GEMM rates remain unmeasured.
- Benches (D2D bandwidth, GEMM rates): NOT RUN - deferred, opt-in only.

## What this evidence establishes (and does not)

Established: SKU/stack/NUMA/interconnect identity (#62 item 1); FP8 variant
semantics on-device; Triton AMD-backend q4 kernel correctness; dtype
roundtrips; and (run 2) the FP8 scaled_mm support matrix — FP8 matmul is
available on this stack with fnuz operands and column-major B, rowwise
scaling for the pure e4m3fnuz pair. NOT established: throughput/SLO of
any kind (benches still not run), multi-GPU collectives/RDMA (#51 cell),
checkpoint serving gates, concurrency, and FP8 accuracy thresholds (the
measured errors carry unit-scale quantization noise; no acceptance
criterion was applied). Per the standing discipline these are evidence
attached to #62/#719, not qualified claims; #62 stays open.

## Run 2 provenance (2026-11-14 16:49-16:50 UTC)

The idle window moved: GPU 0 stayed busy for the full 5-minute wait, so
the run used the healthy idle card at physical PCI `0000:bd:00.0` (host
card41/renderD168, ~23.27 GiB free) through the existing qwen3-omni
container (`7cfabe43636cb0`, `runtime-selection.json`), same containment
(UID/GID 2000, 1 GiB allocator cap, 2 threads, nice 19, private caches,
180 s timeout + 10 s kill grace, 3 consecutive idle samples before
launch). Health identical before/after; kernel log clean. Source scripts
verified byte-identical to repo main (md5). Records:
`RESULTS-run2.local.md`, `execution.py`, `execution-idle-gpu.py`,
`capability-approved-run2.log`, `probes-approved-run2/`.

## Follow-ups

1. Opt-in benches (`--bench`: D2D bandwidth, bf16 + fp8 GEMM rates) —
   still not run; the last unmeasured single-GPU primitive. The fp8 bench
   path's tensorwise-first scaling matches the now-measured support
   matrix.
2. Collectives cell (RCCL over xGMI, 2 GPUs) needs a separate bounded
   multi-rank harness under #51 - not covered by single-GPU probes.
