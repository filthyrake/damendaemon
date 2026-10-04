// Independent Q4_0 matrix-vector products. One warp owns one output channel.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

template<int Rows>
__global__ void q4_gemv(const half* __restrict__ x,
                        const uint8_t* __restrict__ weights,
                        const half* __restrict__ scales,
                        const half* __restrict__ bias,
                        half* __restrict__ output, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (channel >= n) return;
    float sums[Rows] = {};
    const uint8_t* row = weights + static_cast<size_t>(channel) * (k / 2);
    const half* row_scales = scales + static_cast<size_t>(channel) * (k / 32);
    for (int packed_column = lane * 4; packed_column < k / 2; packed_column += 128) {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(row + packed_column);
        const float scale = __half2float(row_scales[packed_column / 16]);
        const int column = (packed_column / 16) * 32 + packed_column % 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t byte = (packed >> (8 * j)) & 255;
            // Preserve the executor's FP16 dequantized-matrix rounding boundary.
            const float low = __half2float(__float2half_rn((static_cast<int>(byte & 15) - 8) * scale));
            const float high = __half2float(__float2half_rn((static_cast<int>(byte >> 4) - 8) * scale));
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                sums[r] = fmaf(low, __half2float(x[r*k + column + j]), sums[r]);
                sums[r] = fmaf(high, __half2float(x[r*k + column + j + 16]), sums[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        for (int offset = 16; offset > 0; offset >>= 1)
            sums[r] += __shfl_down_sync(0xffffffff, sums[r], offset);
        if (lane == 0) {
            const float value = sums[r] + (bias ? __half2float(bias[channel]) : 0.f);
            output[r*n + channel] = __float2half_rn(value);
        }
    }
}

extern "C" int dllm_q4_gemv(const void* x, const void* weights, const void* scales,
                           const void* bias, void* output, int rows, int n, int k,
                           void* stream_pointer) {
    if (rows < 1 || rows > 4 || n <= 0 || k <= 0 || k % 32)
        return static_cast<int>(cudaErrorInvalidValue);
    auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    const dim3 block(128), grid((n + 3) / 4);
    #define LAUNCH(R) q4_gemv<R><<<grid,block,0,stream>>>(static_cast<const half*>(x), \
        static_cast<const uint8_t*>(weights),static_cast<const half*>(scales), \
        static_cast<const half*>(bias),static_cast<half*>(output),n,k)
    switch (rows) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
    }
    #undef LAUNCH
    return static_cast<int>(cudaGetLastError());
}
