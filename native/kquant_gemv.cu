// Independent GGUF K-quant decode. Format reference: MIT-licensed llama.cpp
// gguf-py/gguf/quants.py, 836d57176dc699a726c55418e4f96b8ca628e1bf.
// One warp owns an output row. Keep block headers in registers across values.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Override only for isolated launch-geometry measurements. Normal builds retain
// four warp-owned output rows per block and exactly the same arithmetic.
#ifndef DAMENDAEMON_KQUANT_THREADS
#define DAMENDAEMON_KQUANT_THREADS 128
#endif
static_assert(DAMENDAEMON_KQUANT_THREADS >= 32 && DAMENDAEMON_KQUANT_THREADS <= 1024
              && DAMENDAEMON_KQUANT_THREADS % 32 == 0, "threads must be whole CUDA warps");

template<typename T>
__device__ __forceinline__ T weight_load(const T* pointer) {
#ifdef DAMENDAEMON_KQUANT_BYPASS_L1
    return __ldcg(pointer);
#else
    return *pointer;
#endif
}

template<int Kind, int Rows>
__global__ void kquant_gemv(const half* __restrict__ x,
                          const uint8_t* __restrict__ weights,
                          const half* __restrict__ bias,
                          half* __restrict__ output,
                          int n, int k, int first, int count) {
    constexpr int bytes = Kind == 12 ? 144 : Kind == 13 ? 176 : 210;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= count) return;
    float first_sum[Rows] = {}, second_sum[Rows] = {};
    const uint8_t* row_weights = weights + static_cast<size_t>(row) * (k / 256) * bytes;
    const int packed_column = lane * 4;
    for (int block = 0; block < k / 256; ++block) {
        const uint8_t* data = row_weights + block * bytes;
        const float d = __half2float(__ushort_as_half(weight_load(
            reinterpret_cast<const uint16_t*>(data + (Kind == 14 ? 208 : 0)))));
        const float dm = Kind == 14 ? 0.f : __half2float(__ushort_as_half(weight_load(
            reinterpret_cast<const uint16_t*>(data + 2))));
        // Q6_K rows are only two-byte aligned. Two halfword loads are safe for
        // every segment offset, including a Q4_K segment after an odd Q6_K row.
        const int quant_offset = Kind == 12 ? 16 : Kind == 13 ? 48 : 0;
        const uint8_t* quant = data + quant_offset + packed_column;
        const uint32_t packed = static_cast<uint32_t>(weight_load(reinterpret_cast<const uint16_t*>(quant)))
            | (static_cast<uint32_t>(weight_load(reinterpret_cast<const uint16_t*>(quant + 2))) << 16);
        uint32_t extra = 0;
        if constexpr (Kind != 12) {
            const int extra_offset = Kind == 13 ? 16 + packed_column % 32
                : 128 + (packed_column / 64) * 32 + packed_column % 32;
            const uint8_t* high = data + extra_offset;
            extra = static_cast<uint32_t>(weight_load(reinterpret_cast<const uint16_t*>(high)))
                | (static_cast<uint32_t>(weight_load(reinterpret_cast<const uint16_t*>(high + 2))) << 16);
        }
        #pragma unroll
        for (int side = 0; side < 2; ++side) {
            const int column = Kind == 14 ? (packed_column / 64) * 128 + packed_column % 64 + side * 64
                : (packed_column / 32) * 64 + packed_column % 32 + side * 32;
            float scale, minimum = 0.f;
            int high_shift = 0;
            if constexpr (Kind == 12 || Kind == 13) {
                const int group = column / 32, index = group % 4;
                const int sc = weight_load(data + 4 + index), mn = weight_load(data + 8 + index),
                          shared = weight_load(data + 12 + index);
                scale = __fmul_rn(d, static_cast<float>(group < 4 ? sc & 63 : (shared & 15) | ((sc >> 2) & 48)));
                minimum = __fmul_rn(dm, static_cast<float>(group < 4 ? mn & 63 : (shared >> 4) | ((mn >> 2) & 48)));
                high_shift = group;
            } else {
                scale = __fmul_rn(d, static_cast<float>(static_cast<int8_t>(weight_load(data + 192 + column / 16))));
                high_shift = (column % 128 / 32) * 2;
            }
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                float value[2];
                #pragma unroll
                for (int element = 0; element < 2; ++element) {
                    const int shift = (pair * 2 + element) * 8;
                    int q = (packed >> (shift + side * 4)) & 15;
                    if constexpr (Kind == 13) q |= ((extra >> (shift + high_shift)) & 1) << 4;
                    if constexpr (Kind == 14) q = (q | (((extra >> (shift + high_shift)) & 3) << 4)) - 32;
                    value[element] = __fsub_rn(__fmul_rn(scale, static_cast<float>(q)), minimum);
                }
                const float2 w = __half22float2(__floats2half2_rn(value[0], value[1]));
                #pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    const float2 input = __half22float2(*reinterpret_cast<const half2*>(x + r*k + block*256 + column + pair*2));
                    first_sum[r] = __fmaf_rn(w.x, input.x, first_sum[r]);
                    second_sum[r] = __fmaf_rn(w.y, input.y, second_sum[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float sum = first_sum[r] + second_sum[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0)
            output[r*n + first + row] = __float2half_rn(sum + (bias ? __half2float(bias[first + row]) : 0.f));
    }
}

template<int Kind>
void launch(const half* x, const uint8_t* weights, const half* bias, half* output,
            int rows, int n, int k, int first, int count, cudaStream_t stream) {
    constexpr int warps = DAMENDAEMON_KQUANT_THREADS / 32;
    const dim3 block(DAMENDAEMON_KQUANT_THREADS), grid((count + warps - 1) / warps);
    #define LAUNCH(R) kquant_gemv<Kind,R><<<grid,block,0,stream>>>(x,weights,bias,output,n,k,first,count)
    switch (rows) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
    }
    #undef LAUNCH
}

extern "C" int damendaemon_kquant_gemv(const void* x, const void* weights, const void* bias,
                               void* output, int rows, int n, int k, int first,
                               int count, int kind, void* stream_pointer) {
    if (!x || !weights || !output || rows < 1 || rows > 4 || n <= 0 || k <= 0 || k % 256
        || first < 0 || count <= 0 || first > n || count > n - first || kind < 12 || kind > 14)
        return static_cast<int>(cudaErrorInvalidValue);
    auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    #define DISPATCH(K) launch<K>(static_cast<const half*>(x),static_cast<const uint8_t*>(weights), \
        static_cast<const half*>(bias),static_cast<half*>(output),rows,n,k,first,count,stream)
    switch (kind) {
        case 12: DISPATCH(12); break;
        case 13: DISPATCH(13); break;
        case 14: DISPATCH(14); break;
    }
    #undef DISPATCH
    return static_cast<int>(cudaGetLastError());
}

extern "C" int damendaemon_kquant_abi_version() { return 1; }
