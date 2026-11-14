# MI300 collection completed — 2026-11-14

Inventory and the original capability probes completed. Benchmarks were not run.

The approved run used healthy physical GPU 0 after waiting for an idle window. It ran through standard CRI exec as container UID/GID 2000, with one visible GPU, a 1 GiB PyTorch allocator limit, two CPU threads, nice 19, private caches and a 180-second timeout. Its capability sections ran from 16:17:08 to 16:17:13 UTC (08:17:08–08:17:13 Pacific). No services were restarted or reconfigured.

## Results

- Device: gfx942:sramecc+:xnack-, 304 compute units, 192 GiB HBM. Only the selected GPU was visible to the probe.
- All nine tested dtype cast/roundtrip cases passed, including the four FP8 types.
- E4M3 FN versus FNUZ semantics were measured: encoding of 1.0 is 0x38 versus 0x40; max finite values are 448 versus 240.
- BF16 matmul ran; its measured maximum absolute error versus the FP32 reference was 0.23992156982421875.
- Triton q4 dequantization compiled and passed its reference comparison with maximum absolute error 0.0. TTIR, TTGIR, AMDGCN and HSACO artifacts are included.
- All 32 FP8 scaled_mm cases returned Invalid scaling configuration. The original probe supplies scale_a with shape (256,) and singleton scale_b; those arguments are rejected by this PyTorch runtime. These results do not establish FP8 matmul hardware support or lack of support. The original probe source was not altered to fix this separate issue.
- Exit code was 0. The script catches section/cell failures, so this exit status must be read alongside the cell results above.

## Operational verification

All ten HTTP health responses matched exactly before/after: nine returned 200; port 8403 returned the same pre-existing 404. All observed serving containers remained running with their prior attempt counts. No GPU error/fault/reset lines appeared in the kernel log captured for the approved execution interval. These checks do not establish zero request-latency impact.

Seven healthy GPUs remain visible; quarantined PCI device 0000:9d:00.0 is absent. The host reports ROCm 7.2.1. The existing GLM runtime imports PyTorch 2.12.0+git6bbd260, HIP 7.2.53211 and Triton 3.7.1.

37 host inventory logs are included. Missing tools/unsupported flags are recorded, including amd-smi, numactl, hwloc, hipconfig and hipcc. Some rocm-smi output contains unversioned libdrm_amdgpu.so loader warnings; no host library/package configuration was changed. Original collect.sh Python discovery can misidentify python-config executables; the inventory execution copy explicitly skips host capability execution.

## Execution and artifact notes

Earlier setup failures and logs are retained for provenance. The successful run is capability-approved-run.log; its results and compiled ISA are in collected/evidence/probes-approved/. Earlier root-created mode-0700 scratch was inaccessible to runtime UID 2000; the approved invocation corrected staging by creating scratch through normal runtime exec. The initial approval-review rejection was followed by explicit user approval before the successful invocation.

Original collect.sh, probes.py and README.md remain unchanged. collect-inventory.sh is a copy with an inventory-only gate. The prepared-capability-run.py and prepared-command.txt record the successful bounded invocation and safety checks; they are records, not idempotent rerun scripts.

Working SSH credentials/details were stored and read-back verified in Doppler project damenknight/config dev_personal under MI300_SSH_PASSWORD, MI300_SSH_USER, MI300_SSH_HOST and MI300_SSH_JUMP. No secret values are included in this bundle.

SHA256SUMS.files.txt verifies the final packaged files. The collector's original inner SHA256SUMS.txt includes itself and is retained only as original evidence; use the external manifest for integrity verification. No commits, pushes, PRs or production artifact releases were performed.
