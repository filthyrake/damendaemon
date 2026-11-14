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
- **FP8 `torch._scaled_mm`: UNRESOLVED - probe defect, not a hardware
  verdict.** All 32 dtype x layout cells were rejected with "Invalid
  scaling configuration" because the probe passed a mixed scaling mode
  (per-row `scale_a` shape (256,) with singleton `scale_b`). This build
  requires TensorWise (both scales singleton) or RowWise (both scaled
  per-row/per-col). No conclusion about FP8 matmul support is claimable
  from this run; the corrected probe (commit `~` below) retests with
  tensorwise + rowwise scaling and full error strings.
- Benches (D2D bandwidth, GEMM rates): NOT RUN - deferred, opt-in only.

## What this evidence establishes (and does not)

Established: SKU/stack/NUMA/interconnect identity (#62 item 1); FP8 variant
semantics on-device; Triton AMD-backend q4 kernel correctness; dtype
roundtrips. NOT established: FP8 matmul path (probe bug), throughput/SLO of
any kind (no benches), multi-GPU collectives/RDMA (#51 cell), checkpoint
serving gates, concurrency. Per the standing discipline these are evidence
attached to #62/#719, not qualified claims; #62 stays open.

## Follow-ups

1. Corrected `scaled_mm` probe (tensorwise + rowwise shapes) + opt-in
   benches shipped in `scripts/mi300_collect/` (same commit as this
   evidence) - the next user-executed run closes the FP8-matmul question.
2. Collectives cell (RCCL over xGMI, 2 GPUs) needs a separate bounded
   multi-rank harness under #51 - not covered by single-GPU probes.
