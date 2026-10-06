# AMD ROCm qualification

[Documentation](README.md) · [AMD work and acceptance gates](https://github.com/filthyrake/dllm/issues/52)

AMD is a planned execution target, **not a qualified dllm backend yet**. A ROCm
installation or successful GPU discovery does not establish model correctness,
packed-kernel support, memory safety or competitive inference performance.

## First target: HOMER

Read-only inspection on 2026-10-06 found:

| Property | Observed value |
|---|---|
| Processor / integrated GPU | Ryzen AI Max+ 395 / Radeon 8060S |
| ROCm architecture | `gfx1151` |
| Operating system | Rocky Linux from CIQ Pro AI 9.7 |
| Kernel | `6.19.8-1.el9.elrepo.x86_64` |
| System memory | Approximately 122 GiB total, 111 GiB available at inspection |
| Selected host ROCm path | `/opt/rocm` resolves to `/opt/rocm-7.2.0` |
| Other installed ROCm directories | 7.2.1 and 7.2.4; directory presence is not qualification |
| GPU access | User has access to KFD and render device |

Existing Sheena and Fuzzball services remain active. No service, driver, kernel
or global ROCm setting was changed. The inspection showed 0% GPU activity at
that instant; this is not an exclusive reservation or a benchmark baseline.

`rocm-smi` reported a 2 GiB VRAM aperture while `rocminfo` exposed a much larger
memory pool. Neither reading alone is a safe admission budget for the shared
memory GPU. Allocator behavior, host pressure and actual release must be
qualified before large-model claims. The first work is bounded isolated probes.

## Runtime boundary

PyTorch's HIP build [intentionally uses `torch.cuda` APIs](https://docs.pytorch.org/docs/stable/notes/hip.html).
Those names do not alone make an execution path NVIDIA-only. dllm nevertheless
currently requires NVIDIA-style UUID selection, has NVIDIA-specific native
libraries, and has not qualified its Triton policies on this device. AMD needs
explicit dispatch and capability checks; it must not load CUDA/Marlin binaries.

The [AMD Ryzen Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
identifies gfx1151 support for specified stack combinations. It does not certify
this installed Rocky environment or dllm. Reproducible environment pins and
actual model/performance results will be recorded here as they are obtained.

The task list lives in [issue #52](https://github.com/filthyrake/dllm/issues/52);
the project-wide completion contract is [issue #3](https://github.com/filthyrake/dllm/issues/3).
