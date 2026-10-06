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

## First bounded HIP primitive result

The existing isolated Python 3.11 environment uses PyTorch
`2.11.0+rocm7.13.0a20260425` / HIP `7.13.26162`, independently of the host
`/opt/rocm` symlink. Six real-device matrix, grouped causal attention and
normalization comparisons passed against CPU FP64 arithmetic. Peak Torch
allocation was about 97 MiB. This is not model execution or a speed benchmark.

The [initial report](evidence/amd/homer-hip-primitives-initial.json) failed its
strict release gate: 32 MiB remained after the first operation, stable on the
second cycle. An [explicit workspace-release diagnostic](evidence/amd/homer-hip-primitives-workspace-release.json)
returned allocation to zero after both cycles with unchanged numerical gates.
This identifies the retained allocation as the documented
[HIP BLAS workspace](https://docs.pytorch.org/docs/2.14/notes/hip.html#hipblas-workspaces),
not a retained probe tensor. The diagnostic does not change dllm's runtime
cleanup policy; production admission must account for library workspaces.

The runtime reported roughly 61.5 GiB total device memory but roughly 111 GiB
free through its allocator API. These APU reporting differences still need an
explicit bounded admission policy; no large allocation was attempted. The
attention call used the available default path; experimental fast attention was
not enabled and no fused-attention performance claim is made.

Active model targets are **Qwen3 or newer, with Qwen3.5 prioritized**. Qwen3
dense bring-up and Qwen3.5 hybrid-state correctness are separate gates. Older
fixtures remain historical evidence and cheap regression tests, not new GPU
qualification or tuning targets.

## Qwen3.5 text baseline on the AMD branch

The experimental `codex/amd-rocm-bringup` branch adds explicit `--device rocm`,
HIP build/device checks, rejection of NVIDIA-only libraries and shared-memory
admission. Initial scope is one gfx1151 device. The default process allocation
budget is 8 GiB; admission also limits itself to reported free/total device
memory and preserves 16 GiB of live host headroom. `DLLM_ROCM_MEMORY_GIB` is
an explicit override, not permission to exceed those other bounds. This has
462 passing local regression tests; these tests do not prove GPU speed.

The pinned Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17` passes
[84 actual AMD comparisons](evidence/amd/qwen35-hip-torch-replay512.json) against
a separately generated [CPU FP32 reference](evidence/amd/qwen35-cpu-fp32-golden512.json).
There are no greedy-token mismatches; maximum relative RMS is 0.002173 and
maximum absolute logit error is 0.025072. The gate retains RMS <0.01, absolute
<0.5, finite logits and greedy agreement. It covers 512 context, reordered
batches and slot reuse, using `torch-hybrid` without graph capture.

A reporting-only bug initially interrupted serialization because hybrid
executors do not expose the dense executor's `compute_dtype` attribute. The
replay reporter now records null for that field; numerical checks are unchanged.
This result does not qualify fused kernels, quantized text, images, the HTTP
lifecycle, long context or comparative performance. Those remain issue #52/#53
gates; no old checkpoint is being substituted for the modern target.
