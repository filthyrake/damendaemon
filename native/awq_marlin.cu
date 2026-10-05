// dllm C interface to the separately licensed Marlin device kernels.
// Initial experiment: FP16 activations/output, unsigned AWQ4, group size 128.
#include <algorithm>
#include <type_traits>
#include <limits>
#include <cuda_runtime.h>
#define MARLIN_NAMESPACE_NAME dllm_marlin
#include "marlin_template.h"
#include "awq_repack.cuh"

namespace {
template<bool Small>
constexpr auto kernel = dllm_marlin::Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(),
    128, 1, 8, 4, Small, 4, 8, false>;
}

extern "C" int dllm_awq_marlin_abi() { return 1; }

extern "C" int dllm_awq_marlin_init(int* sms, int* shared) {
    if (!sms || !shared) return cudaErrorInvalidValue;
    int device, major;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) return status;
    if (major < 8) return cudaErrorNotSupported;
    status = cudaDeviceGetAttribute(sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    status = cudaDeviceGetAttribute(shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status != cudaSuccess) return status;
    int budget = *shared/2-1024;
    status = cudaFuncSetAttribute(kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
    if (status != cudaSuccess) return status;
    return cudaFuncSetAttribute(dllm_marlin::awq_marlin_repack_kernel<256, 4, false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, *shared);
}

extern "C" int dllm_awq_marlin_repack(const void* weights, void* packed, int k, int n,
                                      int sms, int shared, void* stream) {
    if (!weights || !packed || k <= 0 || n <= 0 || k%128 || n%128 || sms <= 0 || shared <= 0)
        return cudaErrorInvalidValue;
    dllm_marlin::awq_marlin_repack_kernel<256, 4, false><<<sms, 256, shared, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint32_t*>(weights), static_cast<uint32_t*>(packed), k, n);
    return cudaGetLastError();
}

extern "C" int dllm_awq_marlin_gemm(const void* input, const void* weights,
                                    const void* scales, const void* zeros, const void* bias,
                                    void* output, void* scratch, void* locks,
                                    int m, int n, int k, int sms, int shared, void* stream) {
    if (!input || !weights || !scales || !zeros || !output || !scratch || !locks
            || m < 1 || m > 16 || k <= 0 || n <= 0 || k%128 || n%128 || sms <= 0 || shared <= 0)
        return cudaErrorInvalidValue;
    auto function = m <= 8 ? kernel<true> : kernel<false>;
    const int budget = shared/2-1024;
    function<<<sms*2, 128, budget, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const int4*>(input), static_cast<const int4*>(weights),
        static_cast<int4*>(output), static_cast<int4*>(scratch), static_cast<const int4*>(bias),
        nullptr, static_cast<const int4*>(scales), nullptr, static_cast<const int4*>(zeros),
        m, n, k, k, static_cast<int*>(locks), bias != nullptr, false, true, budget);
    return cudaGetLastError();
}
