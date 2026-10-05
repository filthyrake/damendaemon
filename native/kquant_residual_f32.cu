// Independent K-quant dot products with two dyadic int8 activation terms.
// FP32 input/output variant; ABI 101 is distinct from the FP16 ABI.
// Format reference: MIT-licensed llama.cpp gguf-py/gguf/quants.py,
// 836d57176dc699a726c55418e4f96b8ca628e1bf; independently implemented GPU arithmetic.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float dyadic_scale(float maximum) {
    if (maximum == 0.f) return 0.f;
    const unsigned exponent = __float_as_uint(maximum) & 0x7f800000u;
    // Preserve small FP32 inputs rather than underflowing the exponent bits.
    float scale = exponent < (7u << 23)
        ? ldexpf(1.f, max(-149, ilogbf(maximum) - 6))
        : __uint_as_float(exponent - (6u << 23));
    return maximum > 127.f * scale ? scale * 2.f : scale;
}

__global__ void quantize(const float* x, int8_t* q1, int8_t* q2, float4* scales, int groups) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= groups) return;
    const int index = group * 32 + lane;
    const float value = x[index];
    float maximum = fabsf(value), sum = value;
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    const float d1 = dyadic_scale(maximum);
    const int first = d1 == 0.f ? 0 : max(-127, min(127, __float2int_rn(value / d1)));
    const float residual = __fmaf_rn(-static_cast<float>(first), d1, value);
    maximum = fabsf(residual);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    const float d2 = dyadic_scale(maximum);
    const int second = d2 == 0.f ? 0 : max(-127, min(127, __float2int_rn(residual / d2)));
    q1[index] = static_cast<int8_t>(first);q2[index] = static_cast<int8_t>(second);
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) scales[group] = make_float4(d1, d2, sum, 0.f);
}

__device__ __forceinline__ uint32_t load_word(const uint8_t* pointer) {
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(pointer))
        | (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(pointer + 2)) << 16);
}

template<int Kind, int Rows>
__global__ void multiply(const int8_t* q1, const int8_t* q2, const float4* scales,
                         const uint8_t* weights, const float* bias, float* output,
                         int n, int k, int first, int count) {
    constexpr int bytes = Kind == 12 ? 144 : Kind == 13 ? 176 : 210;
    constexpr int subgroup = Kind == 14 ? 4 : 8;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= count) return;
    const uint8_t* row_weights = weights + static_cast<size_t>(row) * (k / 256) * bytes;
    const int packed_column = lane * 4;
    float accum[Rows] = {};
    for (int block = 0; block < k / 256; ++block) {
        const uint8_t* data = row_weights + block * bytes;
        const float d = __half2float(*reinterpret_cast<const half*>(data + (Kind == 14 ? 208 : 0)));
        const float dm = Kind == 14 ? 0.f : __half2float(*reinterpret_cast<const half*>(data + 2));
        const uint32_t packed = load_word(data + (Kind == 12 ? 16 : Kind == 13 ? 48 : 0) + packed_column);
        uint32_t extra = 0;
        if constexpr (Kind != 12)
            extra = load_word(data + (Kind == 13 ? 16 + packed_column % 32
                : 128 + (packed_column / 64) * 32 + packed_column % 32));
        #pragma unroll
        for (int side = 0; side < 2; ++side) {
            const int column = Kind == 14 ? (packed_column / 64) * 128 + packed_column % 64 + side * 64
                : (packed_column / 32) * 64 + packed_column % 32 + side * 32;
            uint32_t values = (packed >> (side * 4)) & 0x0f0f0f0fu;
            float scale, minimum = 0.f;
            if constexpr (Kind == 12 || Kind == 13) {
                const int group = column / 32, index = group % 4;
                const int sc = data[4 + index], mn = data[8 + index], shared = data[12 + index];
                scale = __fmul_rn(d, static_cast<float>(group < 4 ? sc & 63 : (shared & 15) | ((sc >> 2) & 48)));
                minimum = __fmul_rn(dm, static_cast<float>(group < 4 ? mn & 63 : (shared >> 4) | ((mn >> 2) & 48)));
                if constexpr (Kind == 13) values |= ((extra >> group) & 0x01010101u) << 4;
            } else {
                values |= ((extra >> ((column % 128 / 32) * 2)) & 0x03030303u) << 4;
                // Signed six-bit coefficients, extended independently per byte.
                values = (values & 0x1f1f1f1fu) | ((~values & 0x20202020u) * 7u);
                scale = __fmul_rn(d, static_cast<float>(static_cast<int8_t>(data[192 + column / 16])));
            }
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int index = r*k + block*256 + column;
                int a = __dp4a(static_cast<int>(values), *reinterpret_cast<const int*>(q1 + index), 0);
                int b = __dp4a(static_cast<int>(values), *reinterpret_cast<const int*>(q2 + index), 0);
                const float4 activation = scales[index / 32];
                const float dot = __fadd_rn(__fmul_rn(static_cast<float>(a), activation.x),
                                           __fmul_rn(static_cast<float>(b), activation.y));
                float value = __fmul_rn(scale, dot);
                // Scale each lane's four-term partial directly. Only the
                // affine minimum needs one contribution per 32-value group.
                // Avoid repeated subgroup shuffles inside the reduction loop.
                if constexpr (Kind != 14)
                    if (lane % subgroup == 0) value = __fsub_rn(value, __fmul_rn(minimum, activation.z));
                accum[r] += value;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        float sum = accum[r];
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) output[r*n + first + row] = (sum + (bias ? bias[first+row] : 0.f));
    }
}

template<int Kind>
void launch(const int8_t* q1, const int8_t* q2, const float4* scales, const uint8_t* weights,
            const float* bias, float* output, int rows, int n, int k, int first, int count, cudaStream_t stream) {
    const dim3 block(128), grid((count+3)/4);
    #define LAUNCH(R) multiply<Kind,R><<<grid,block,0,stream>>>(q1,q2,scales,weights,bias,output,n,k,first,count)
    switch(rows) {case 1: LAUNCH(1);break;case 2: LAUNCH(2);break;case 3: LAUNCH(3);break;case 4: LAUNCH(4);break;}
    #undef LAUNCH
}

extern "C" int dllm_residual_quantize(const void* x, void* q1, void* q2, void* scales, int rows, int k, void* stream) {
    if (!x || !q1 || !q2 || !scales || rows<1 || rows>4 || k<=0 || k%256) return cudaErrorInvalidValue;
    const int groups=rows*(k/32);
    quantize<<<(groups+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(x),static_cast<int8_t*>(q1),static_cast<int8_t*>(q2),static_cast<float4*>(scales),groups);
    return cudaGetLastError();
}

extern "C" int dllm_residual_gemv(const void* q1, const void* q2, const void* scales, const void* weights,
                                  const void* bias, void* output, int rows, int n, int k,
                                  int first, int count, int kind, void* stream_pointer) {
    if (!q1 || !q2 || !scales || !weights || !output || rows<1 || rows>4 || n<=0 || k<=0 || k%256
        || first<0 || count<=0 || first>n || count>n-first || kind<12 || kind>14) return cudaErrorInvalidValue;
    auto stream=reinterpret_cast<cudaStream_t>(stream_pointer);
    #define DISPATCH(K) launch<K>(static_cast<const int8_t*>(q1),static_cast<const int8_t*>(q2), \
        static_cast<const float4*>(scales),static_cast<const uint8_t*>(weights),static_cast<const float*>(bias), \
        static_cast<float*>(output),rows,n,k,first,count,stream)
    switch(kind) {case 12: DISPATCH(12);break;case 13: DISPATCH(13);break;case 14: DISPATCH(14);break;}
    #undef DISPATCH
    return cudaGetLastError();
}

extern "C" int dllm_residual_abi_version() { return 101; }
