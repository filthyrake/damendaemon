// Hardware characterization only: this is not an LLM throughput benchmark.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

struct DeviceBuffer {
    void *ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { check(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};
struct HostBuffer {
    void *ptr = nullptr;
    explicit HostBuffer(size_t bytes) { check(cudaMallocHost(&ptr, bytes)); }
    ~HostBuffer() { if (ptr) cudaFreeHost(ptr); }
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;
};
struct Blas {
    cublasHandle_t handle{};
    Blas() { check(cublasCreate(&handle)); }
    ~Blas() { cublasDestroy(handle); }
};
struct Timer {
    cudaEvent_t start{}, end{};
    Timer() {
        check(cudaEventCreate(&start));
        try { check(cudaEventCreate(&end)); }
        catch (...) { cudaEventDestroy(start); throw; }
    }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(end); }
    void begin() { check(cudaEventRecord(start)); }
    float finish() {
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float ms;
        check(cudaEventElapsedTime(&ms, start, end));
        return ms;
    }
};

__global__ void fill_half(__half *data, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) data[i] = __float2half(0.01f);
}

static void matmul(cublasHandle_t handle, int batch, bool comma) {
    constexpr int m = 4096, k = 4096, iterations = 100;
    DeviceBuffer a(size_t(m) * k * sizeof(__half));
    DeviceBuffer b(size_t(k) * batch * sizeof(__half));
    DeviceBuffer c(size_t(m) * batch * sizeof(__half));
    fill_half<<<(size_t(m) * k + 255) / 256, 256>>>((__half *)a.ptr, size_t(m) * k);
    fill_half<<<(size_t(k) * batch + 255) / 256, 256>>>((__half *)b.ptr, size_t(k) * batch);
    check(cudaGetLastError());
    float alpha = 1.f, beta = 0.f;
    auto run = [&] {
        check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, batch, k,
                          &alpha, a.ptr, CUDA_R_16F, m, b.ptr, CUDA_R_16F, k,
                          &beta, c.ptr, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    for (int i = 0; i < 10; ++i) run();
    check(cudaDeviceSynchronize());
    Timer timer;
    timer.begin();
    for (int i = 0; i < iterations; ++i) run();
    float ms = timer.finish() / iterations;
    __half sample;
    check(cudaMemcpy(&sample, c.ptr, sizeof(sample), cudaMemcpyDeviceToHost));
    const float value = __half2float(__float2half(0.01f));
    const float expected = k * value * value;
    if (std::fabs(__half2float(sample) - expected) > 0.002f)
        throw std::runtime_error("matrix output validation failed");
    std::printf("%s{\"m\":%d,\"n\":%d,\"k\":%d,\"mean_ms\":%.6f,"
                "\"effective_tflops\":%.6f,\"sample_validated\":true}",
                comma ? "," : "", m, batch, k, ms,
                2.0 * m * batch * k / (ms * 1e9));
}

int main() {
    try {
        // Require explicit isolation. CUDA runtime enumerates only this UUID.
        const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!visible || std::string(visible).rfind("GPU-", 0) != 0 ||
            std::string(visible).find(',') != std::string::npos)
            throw std::runtime_error("set CUDA_VISIBLE_DEVICES to exactly one GPU UUID");
        int count;
        check(cudaGetDeviceCount(&count));
        if (count != 1) throw std::runtime_error("expected exactly one visible GPU");
        check(cudaSetDevice(0));
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, 0));
        size_t free_bytes, total_bytes;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        constexpr size_t bytes = 256ull * 1024 * 1024;
        if (free_bytes < 1024ull * 1024 * 1024)
            throw std::runtime_error("less than 1 GiB free; refusing probe");
        int driver, runtime;
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        HostBuffer host(bytes);
        DeviceBuffer device(bytes);
        auto data = static_cast<unsigned char *>(host.ptr);
        for (size_t i = 0; i < bytes; ++i) data[i] = static_cast<unsigned char>(i % 251);
        check(cudaMemcpy(device.ptr, host.ptr, bytes, cudaMemcpyHostToDevice));
        Timer timer;
        timer.begin();
        for (int i = 0; i < 10; ++i)
            check(cudaMemcpyAsync(device.ptr, host.ptr, bytes, cudaMemcpyHostToDevice));
        double ms = timer.finish() / 10;
        // Validate copied data, outside the measured interval.
        for (size_t i = 0; i < bytes; ++i) data[i] = 0;
        check(cudaMemcpy(host.ptr, device.ptr, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < bytes; ++i)
            if (data[i] != static_cast<unsigned char>(i % 251))
                throw std::runtime_error("transfer validation failed");
        std::printf("{\"schema_version\":1,\"kind\":\"synthetic_cuda_probe\","
                    "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
                    "\"driver_version\":%d,\"runtime_version\":%d,"
                    "\"total_bytes\":%zu,\"free_bytes_before\":%zu,"
                    "\"h2d_bytes\":%zu,\"h2d_mean_ms\":%.6f,"
                    "\"h2d_decimal_gb_per_second\":%.6f,\"fp16_gemm\":[",
                    prop.name, prop.major, prop.minor, driver, runtime,
                    total_bytes, free_bytes, bytes, ms, bytes / (ms * 1e6));
        Blas blas;
        matmul(blas.handle, 1, false);
        matmul(blas.handle, 8, true);
        matmul(blas.handle, 32, true);
        std::printf("]}\n");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "dllm CUDA probe: %s\n", error.what());
        return 1;
    }
}
