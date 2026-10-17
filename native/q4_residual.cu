// Independent single-row Q4_0 integer dots with two dyadic int8 activation terms.
// Preserve packed source coefficients; scale in FP32 rather than rounding each
// coefficient to FP16. Explicitly selected only for the qualified model scope.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float dyadic_scale(float maximum) {
    if (maximum == 0.f) return 0.f;
    const unsigned exponent = __float_as_uint(maximum) & 0x7f800000u;
    float scale = __uint_as_float(exponent - (6u << 23));
    return maximum > 127.f * scale ? scale * 2.f : scale;
}

__global__ void quantize(const half* x, int8_t* q1, int8_t* q2, float4* scales, int groups) {
    const int lane = threadIdx.x & 31;
    const int group = blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= groups) return;
    const int index = group * 32 + lane;
    const float value = __half2float(x[index]);
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
    // ABI 202 pairs each low/high nibble dot's four activation bytes. Each
    // dyadic term still occupies K bytes and retains exactly the same values.
    unsigned packed_first = (static_cast<unsigned>(first)&255u) << ((lane&3)*8);
    unsigned packed_second = (static_cast<unsigned>(second)&255u) << ((lane&3)*8);
    packed_first |= __shfl_xor_sync(0xffffffff, packed_first, 1);
    packed_first |= __shfl_xor_sync(0xffffffff, packed_first, 2);
    packed_second |= __shfl_xor_sync(0xffffffff, packed_second, 1);
    packed_second |= __shfl_xor_sync(0xffffffff, packed_second, 2);
    const unsigned high_first = __shfl_xor_sync(0xffffffff, packed_first, 16);
    const unsigned high_second = __shfl_xor_sync(0xffffffff, packed_second, 16);
    if (lane<16 && (lane&3)==0) {
        reinterpret_cast<uint2*>(q1)[group*4+lane/4] = make_uint2(packed_first, high_first);
        reinterpret_cast<uint2*>(q2)[group*4+lane/4] = make_uint2(packed_second, high_second);
    }
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) scales[group] = make_float4(d1, d2, sum, 0.f);
}

template<int Rows, bool StreamWeights>
__global__ void q4_residual(const int8_t* q1, const int8_t* q2, const float4* xs,
                          const uint8_t* w, const half* ws, const half* bias,
                          half* output, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x*4 + threadIdx.x/32;
    if (row >= n) return;
    const uint8_t* weights = w + static_cast<size_t>(row)*(k/2);
    const half* weight_scales = ws + static_cast<size_t>(row)*(k/32);
    float accum[Rows] = {};
    for (int packed_column=lane*4; packed_column<k/2; packed_column+=128) {
        uint32_t packed;
        if constexpr (StreamWeights)
            packed=__ldcg(reinterpret_cast<const uint32_t*>(weights+packed_column));
        else packed=*reinterpret_cast<const uint32_t*>(weights+packed_column);

        uint32_t lo=packed & 0x0f0f0f0fu, hi=(packed>>4) & 0x0f0f0f0fu;
        lo=(lo & 0x07070707u) | ((~lo & 0x08080808u)*31u);
        hi=(hi & 0x07070707u) | ((~hi & 0x08080808u)*31u);
        const int column=(packed_column/16)*32+packed_column%16;
        const float scale=__half2float(weight_scales[packed_column/16]);

        #pragma unroll
        for (int r=0; r<Rows; ++r) {
            const int index=r*k+column;
            const uint2 first_pair=*reinterpret_cast<const uint2*>(q1+r*k+2*packed_column);
            const uint2 second_pair=*reinterpret_cast<const uint2*>(q2+r*k+2*packed_column);
            const int a=__dp4a(static_cast<int>(lo),static_cast<int>(first_pair.x),
                        __dp4a(static_cast<int>(hi),static_cast<int>(first_pair.y),0));
            const int b=__dp4a(static_cast<int>(lo),static_cast<int>(second_pair.x),
                        __dp4a(static_cast<int>(hi),static_cast<int>(second_pair.y),0));
            const float4 d=xs[index/32];
            accum[r]+=scale*(static_cast<float>(a)*d.x+static_cast<float>(b)*d.y);
        }
    }
    #pragma unroll
    for (int r=0; r<Rows; ++r) {
        float sum=accum[r];
        for (int offset=16;offset;offset>>=1) sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if (lane==0) output[r*n+row]=__float2half_rn(sum+(bias?__half2float(bias[row]):0.f));
    }
}

extern "C" int damendaemon_q4_residual_quantize(const void* x, void* q1, void* q2, void* scales,
                                         int rows, int k, void* stream) {
    if (!x || !q1 || !q2 || !scales || rows!=1 || k<=0 || k%256) return cudaErrorInvalidValue;
    const int groups=k/32;
    quantize<<<(groups+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const half*>(x),static_cast<int8_t*>(q1),static_cast<int8_t*>(q2),static_cast<float4*>(scales),groups);
    return cudaGetLastError();
}

extern "C" int damendaemon_q4_residual_gemv(const void* q1,const void* q2,const void* xs,
    const void* w,const void* ws,const void* bias,void* output,int rows,int n,int k,void* stream) {
    if (!q1 || !q2 || !xs || !w || !ws || !output || rows!=1 || n<=0 || k<=0 || k%256)
        return cudaErrorInvalidValue;
    // The qualified wide down projection benefits from bypassing L1. Other
    // measured projections retain ordinary caching; dispatch once per launch.
    if (n==4096 && k==14336) {
        q4_residual<1,true><<<(n+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const int8_t*>(q1),static_cast<const int8_t*>(q2),static_cast<const float4*>(xs),
            static_cast<const uint8_t*>(w),static_cast<const half*>(ws),static_cast<const half*>(bias),
            static_cast<half*>(output),n,k);
    } else {
        q4_residual<1,false><<<(n+3)/4,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(
            static_cast<const int8_t*>(q1),static_cast<const int8_t*>(q2),static_cast<const float4*>(xs),
            static_cast<const uint8_t*>(w),static_cast<const half*>(ws),static_cast<const half*>(bias),
            static_cast<half*>(output),n,k);
    }
    return cudaGetLastError();
}

extern "C" int damendaemon_q4_residual_abi_version() { return 202; }
