# Initial NVIDIA A2 characterization

Measured 2026-10-03 on the user-designated lab VM. The probe selected only the
A2 by UUID. No existing service, container, driver, or device setting was changed.
Source and executable were placed in `~/dllm-dev/` on the test host.

| Property | Observed |
| --- | --- |
| Device | NVIDIA A2, compute capability 8.6 |
| CUDA-visible total memory | 15,691,415,552 bytes |
| CUDA toolkit used to compile | 12.6, nvcc V12.6.85 |
| Runtime / driver API versions | 12060 / 13020 |
| PCIe link at inspection | Gen 3, x8 |
| Pinned host-to-device copy | 256 MiB, ten repetitions after one warmup |
| Mean copy latency | 42.961388 ms |
| Effective transfer bandwidth | 6.248296 GB/s (decimal) |

FP16 inputs/output, FP32 compute, cuBLAS default tensor-op algorithm, column-major
4096 x 4096 weight matrix. Each shape uses ten warmups and 100 timed operations.

| Batch columns | Mean operation latency | Effective TFLOP/s |
| --- | --- | --- |
| 1 | 0.227451 ms | 0.147524 |
| 8 | 0.214812 ms | 1.249631 |
| 32 | 0.202967 ms | 5.290228 |

Every transferred byte was checked after the timing window. One output element
per GEMM shape was checked against its known constant-input result. This is a
smoke validation, not full numerical validation across model operations.

At this measured transfer rate, 8 GiB / 6.248296 GB/s is approximately 1.375
seconds. That is an illustrative transfer-only estimate, **not** an observed
8 GiB model load. Allocation, repacking, disk reads, graph preparation, and
first inference add costs. Small-transfer bandwidth may not predict a full
model accurately.

The existing inference process held a small A2 context before the probe, and the
VM also hosts other workloads. These are initial shared-host measurements, not
controlled comparisons or peak hardware claims. No per-token LLM performance,
model-switch latency, quantization speed, or vLLM/llama.cpp comparison has been
measured yet. Steady-state repeated GEMMs omit real model layer transitions,
attention, KV traffic, and end-to-end scheduling. The A30 was not selected.

Compilation used the direct nvcc command documented in the README. The CMake
configuration is supplied for normal development but was not used for this run.
