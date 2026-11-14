# MI300X evidence collection (issues #62, #719)

Bounded, read-only probes for qualifying an MI300-class datacenter host
independently of HOMER (#62) and for the ROCm side of the CUDA+ROCm fleet
split (#719). Nothing here changes system state; GPU memory is allocated
only on the single `--gpu` index, and only for capability checks and
opt-in benches.

## What to run

```bash
chmod +x collect.sh
./collect.sh                # read-only inventory + capability probes
./collect.sh --bench        # + time-boxed bandwidth/GEMM microbenches (~2 min)
./collect.sh --bench --gpu 3   # pick which GPU the probes/benches use
```

If the torch env isn't on the default `python3`, point at it:

```bash
MI300_PYTHON=/opt/conda/envs/rocm/bin/python ./collect.sh --bench
```

Send back the printed `.tar.gz`. If something stalls, individual logs are
in the evidence dir — partial bundles still beat nothing (every phase
writes as it goes, `probes.py` rewrites `probe_results.json` after each
section).

## What each phase collects and why

| Phase | Evidence | Feeds |
|---|---|---|
| A logs | `rocminfo`, `rocm-smi`/`amd-smi` (hw, product, driver, topo, pids, mem, power, clocks), `hipconfig --full`, `lscpu`/`numactl -H` (NUMA), `lspci`, `/sys/class/kfd`, ldconfig ROCm libs (RCCL/hipBLASLt/MIOpen/hip), kernel module version | #62 "pin actual SKU, memory/NUMA/interconnect, ROCm/HIP stack"; fleet-fit arithmetic (VRAM + host RAM inventory) |
| B probes | device props for ALL GPUs (`gcnArchName` — expect `gfx942`), dtype scan, **FP8 fn vs fnuz byte-level decode tables** (1.0 = 0x38 vs 0x40, max 448 vs 240), `torch._scaled_mm` dtype/layout support matrix with error strings, **Triton q4-dequant kernel**: compile + numeric check vs plain-torch reference + `.amdgcn`/`.hsaco` ISA dump | #62 "capability-based backend selection, primitive arithmetic, FP8/packed formats"; #719 GLM q4 kernels on ROCm; the CUDA→ROCm FP8-variant boundary |
| C benches (opt-in) | HBM D2D copy GB/s (read+write), bf16 8192³ GEMM TFLOPs, fp8 scaled_mm TFLOPs — sized to free VRAM, hard wall-clock budget | #62 "no unmeasured extrapolation"; #54 scorecard cells |

Expected-ish MI300X shapes (for sanity-checking the returned evidence, not
gates): ~192 GiB HBM3 per GPU, >3 TB/s D2D, gfx942, RCCL for collectives.
The probes never assume these — they measure.

## Notes

- `rocm-smi` may be deprecated on your stack in favor of `amd-smi`; both
  are captured, failures recorded per-command.
- The Triton ISA capture copies recently-touched cache entries (`.amdgcn`,
  `.hsaco`, `.ttir`) — if your `TRITON_CACHE_DIR` is elsewhere it's picked
  up automatically.
- Multi-GPU collectives (RCCL all-reduce, P2P over xGMI) are NOT probed
  here — that needs a later two-rank cell under #51 once single-GPU
  capability is established.
