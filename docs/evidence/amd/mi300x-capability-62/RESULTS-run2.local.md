# Updated MI300 collector — completed 2026-11-14

The updated host inventory and original capability script completed with exit code 0. Benchmarks were not enabled. Original source files were preserved and verified byte-for-byte against /Volumes/MainPool/mi300_collect at packaging time.

## Execution

GPU 0 remained busy for the five-minute idle wait; no kernels were launched on that card. A read-only fleet inspection found the healthy physical GPU at 0000:bd:00.0 (host card41, renderD168) idle with about 23.27 GiB free. The run used its existing qwen3-omni container 7cfabe43636cb0 through standard CRI exec as UID/GID 2000. It exposed exactly one GPU, numbered 0 inside the runtime. The quarantined 0000:9d:00.0 card remained absent.

Limits: a 1 GiB PyTorch allocator budget, two CPU threads, nice 19, isolated runtime-owned scratch/cache directories, and a 180-second timeout plus 10-second kill grace. The GPU was verified idle for three consecutive samples before launch. No serving process was stopped, restarted or reconfigured.

Capability sections ran from 2026-11-14T16:49:49Z to 2026-11-14T16:50:21Z (08:49:49–08:50:21 Pacific). Runtime: PyTorch 2.12.0+git6bbd260, HIP 7.2.53211, Triton 3.7.1. Host ROCm inventory remains separate from the container runtime.

## Results

- gfx942:sramecc+:xnack-, 304 compute units, 192 GiB HBM. The probe saw exactly one device.
- All nine dtype GPU casts/roundtrips completed.
- FP8 byte-level semantics and decode tables were collected.
- BF16 matmul executed; max absolute error versus FP32: 0.23197555541992188.
- Triton q4 dequantization compiled and passed its reference comparison with maximum absolute error 0.0. TTIR, TTGIR, AMDGCN and HSACO files are included.
- 5 of 64 FP8 dtype/layout/scaling configurations executed successfully. The other 59 returned explicit layout, dtype, HIPBLAS support or solution-selection errors. No case returned the previous Invalid scaling configuration error.

Successful FP8 cases (tw = tensorwise, rw = rowwise; col denotes column-major B):

| Case | Max absolute error versus unquantized FP32 |
| --- | ---: |
| e4m3fnuzxe4m3fnuz:col:tw | 2.9153 |
| e4m3fnuzxe4m3fnuz:col:rw | 2.9153 |
| e4m3fnuzxe5m2fnuz:col:tw | 4.4535 |
| e5m2fnuzxe4m3fnuz:col:tw | 4.3172 |
| e5m2fnuzxe5m2fnuz:col:tw | 6.2373 |

These successful calls demonstrate operation availability for the tested dimensions and arguments. Their error values include FP8 quantization; the collector applies no accuracy acceptance threshold to them. They are not general correctness or throughput qualification.

Failure counts:

- 20: `RuntimeError: Only multiplication of row-major and column-major matrices is supported by cuBLASLt`
- 8: `ValueError: Expected b.dtype() == at::kFloat8_e4m3fnuz, got: c10::Float8_e4m3fn`
- 14: `RuntimeError: CUDA error: HIPBLAS_STATUS_NOT_SUPPORTED when calling `HIPBLAS_STATUS_NOT_SUPPORTED``
- 8: `ValueError: Expected b.dtype() == at::kFloat8_e4m3fnuz, got: c10::Float8_e5m2`
- 8: `ValueError: Expected b.dtype() == at::kFloat8_e4m3fnuz, got: c10::Float8_e5m2fnuz`
- 1: `RuntimeError: could not find valid hipblaslt solution`

## Operational verification and artifacts

All ten HTTP health responses were identical before/after: nine returned 200; port 8403 returned its pre-existing 404. No GPU error/fault/reset lines appeared in the kernel log captured for the execution interval. This is evidence of unchanged sampled health, not proof of zero request-latency impact.

Host inventory logs preserve missing-tool and unsupported-flag failures. The copied collect-inventory.sh adds the same inventory-only gate used previously; the updated original collect.sh and probes.py are included unchanged. The execution files record both the GPU 0 idle-wait attempt and the completed idle-card run. Credentials were retrieved from Doppler and are not included in the artifacts.

Use SHA256SUMS.files.txt to verify the packaged files. The collector's inner SHA256SUMS.txt includes itself and is retained only as original evidence. Nothing was committed, pushed or published.
