// Independent Q4_0 matrix-vector products. One warp owns one output channel.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ half2 half2_bits(uint32_t bits) {
    __half2_raw raw;
    raw.x = static_cast<unsigned short>(bits);
    raw.y = static_cast<unsigned short>(bits >> 16);
    return half2(raw);
}

__device__ __forceinline__ float2 dequant_pair(uint32_t packed, int shift, half2 scale) {
    // At 1024, adjacent FP16 encodings differ by exactly one. Construct
    // [1024+n0,1024+n1], subtract 1032, then round the scaled result to FP16.
    const uint32_t bits = 0x64006400u | ((packed >> shift) & 15u)
        | (((packed >> (shift + 8)) & 15u) << 16);
    return __half22float2(__hmul2(__hsub2(half2_bits(bits), half2_bits(0x64086408u)), scale));
}

template<int Rows>
__global__ void q4_gemv(const half* __restrict__ x,
                        const uint8_t* __restrict__ weights,
                        const half* __restrict__ scales,
                        const half* __restrict__ bias,
                        half* __restrict__ output, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int channel = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (channel >= n) return;
    float first[Rows] = {}, second[Rows] = {};
    const uint8_t* row = weights + static_cast<size_t>(channel) * (k / 2);
    const half* row_scales = scales + static_cast<size_t>(channel) * (k / 32);
    for (int packed_column = lane * 4; packed_column < k / 2; packed_column += 128) {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(row + packed_column);
        const half2 scale = __half2half2(row_scales[packed_column / 16]);
        const int column = (packed_column / 16) * 32 + packed_column % 16;
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const float2 low = dequant_pair(packed, pair * 16, scale);
            const float2 high = dequant_pair(packed, pair * 16 + 4, scale);
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const float2 xl = __half22float2(*reinterpret_cast<const half2*>(x + r*k + column + pair*2));
                const float2 xh = __half22float2(*reinterpret_cast<const half2*>(x + r*k + column + pair*2 + 16));
                first[r] = fmaf(low.x, xl.x, first[r]);
                second[r] = fmaf(low.y, xl.y, second[r]);
                first[r] = fmaf(high.x, xh.x, first[r]);
                second[r] = fmaf(high.y, xh.y, second[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float sum = first[r] + second[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            const float value = sum + (bias ? __half2float(bias[channel]) : 0.f);
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

extern "C" int dllm_q4_abi_version() { return 1; }
