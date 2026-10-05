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
template<int Blocks, bool Small = false>
constexpr auto kernel = dllm_marlin::Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(),
    128, Blocks, 8, 4, Small, 4, 8, false>;
#ifdef DLLM_AWQ_EXPERIMENT_WIDE
// Isolated tuning build: leave the qualified default and prompt kernels intact.
constexpr auto wide_decode = dllm_marlin::Marlin<
    vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(),
    256, 1, 8, 8, true, 4, 8, false>;
#endif
}

extern "C" int dllm_awq_marlin_abi() { return 2; }

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
#ifdef DLLM_AWQ_EXPERIMENT_WIDE
    status = cudaFuncSetAttribute(wide_decode, cudaFuncAttributeMaxDynamicSharedMemorySize, *shared);
    if (status != cudaSuccess) return status;
#endif
    status = cudaFuncSetAttribute(kernel<1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, budget);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, *shared);
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
            || m < 1 || k <= 0 || n <= 0 || k%128 || n%128 || sms <= 0 || shared <= 0)
        return cudaErrorInvalidValue;
    // The device kernel handles parallel full M tiles. Keep a partial tail in
    // its own launch so floor division cannot discard rows beyond a full tile.
    const half* a = static_cast<const half*>(input);
    half* c = static_cast<half*>(output);
    while (m > 0) {
        const int rows = m > 64 ? std::min((m/64)*64, 1024) : m;
        const int blocks_m = rows > 32 ? 4 : rows > 16 ? 2 : 1;
        auto function = rows <= 8 ? kernel<1, true> :
                        blocks_m == 1 ? kernel<1> : blocks_m == 2 ? kernel<2> : kernel<4>;
        int budget = blocks_m == 4 ? shared : shared/2-1024;
        int blocks = blocks_m == 4 ? sms : sms*2;
        int threads = 128;
#ifdef DLLM_AWQ_EXPERIMENT_WIDE
        if (rows <= 8) {
            function = wide_decode;
            budget = shared;
            blocks = sms;
            threads = 256;
        }
#endif
        function<<<blocks, threads, budget, static_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const int4*>(a), static_cast<const int4*>(weights),
            reinterpret_cast<int4*>(c), static_cast<int4*>(scratch), static_cast<const int4*>(bias),
            nullptr, static_cast<const int4*>(scales), nullptr, static_cast<const int4*>(zeros),
            rows, n, k, k, static_cast<int*>(locks), bias != nullptr, false, true, budget);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        a += static_cast<size_t>(rows)*k;
        c += static_cast<size_t>(rows)*n;
        m -= rows;
    }
    return cudaSuccess;
}
