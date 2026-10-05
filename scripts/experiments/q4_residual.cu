// Validation-only Q4_0 integer dot products. Reuse dllm's own two-term
// activation quantizer; source quantized bytes and scales remain unchanged.
// Unlike native/q4_gemv.cu, multiplication applies the FP16 scale in FP32
// without rounding each coefficient to FP16. This arithmetic is experimental.
#include "../../native/kquant_residual.cu"

template<int Rows>
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
#ifdef DLLM_Q4_RESIDUAL_BYPASS_L1
        // Weight bytes are consumed once; keep the small reused activation
        // arrays in L1 instead. This variant is measured independently.
        const uint32_t packed=__ldcg(reinterpret_cast<const uint32_t*>(weights+packed_column));
#else
        const uint32_t packed=*reinterpret_cast<const uint32_t*>(weights+packed_column);
#endif
        uint32_t lo=packed & 0x0f0f0f0fu, hi=(packed>>4) & 0x0f0f0f0fu;
        lo=(lo & 0x07070707u) | ((~lo & 0x08080808u)*31u);
        hi=(hi & 0x07070707u) | ((~hi & 0x08080808u)*31u);
        const int column=(packed_column/16)*32+packed_column%16;
#ifdef DLLM_Q4_RESIDUAL_BYPASS_L1
        const float scale=__half2float(__ushort_as_half(__ldcg(
            reinterpret_cast<const unsigned short*>(weight_scales+packed_column/16))));
#else
        const float scale=__half2float(weight_scales[packed_column/16]);
#endif
        #pragma unroll
        for (int r=0; r<Rows; ++r) {
            const int index=r*k+column;
            const int a=__dp4a(static_cast<int>(lo),*reinterpret_cast<const int*>(q1+index),
                        __dp4a(static_cast<int>(hi),*reinterpret_cast<const int*>(q1+index+16),0));
            const int b=__dp4a(static_cast<int>(lo),*reinterpret_cast<const int*>(q2+index),
                        __dp4a(static_cast<int>(hi),*reinterpret_cast<const int*>(q2+index+16),0));
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

extern "C" int dllm_q4_residual_gemv(const void* q1,const void* q2,const void* xs,
    const void* w,const void* ws,const void* bias,void* output,int rows,int n,int k,void* stream) {
    if (!q1 || !q2 || !xs || !w || !ws || !output || rows<1 || rows>4 || n<=0 || k<=0 || k%256)
        return cudaErrorInvalidValue;
    const dim3 block(128),grid((n+3)/4);
    #define Q4_LAUNCH(R) q4_residual<R><<<grid,block,0,reinterpret_cast<cudaStream_t>(stream)>>>( \
        static_cast<const int8_t*>(q1),static_cast<const int8_t*>(q2),static_cast<const float4*>(xs), \
        static_cast<const uint8_t*>(w),static_cast<const half*>(ws),static_cast<const half*>(bias), \
        static_cast<half*>(output),n,k)
    switch(rows) {case 1: Q4_LAUNCH(1);break;case 2: Q4_LAUNCH(2);break;case 3: Q4_LAUNCH(3);break;case 4: Q4_LAUNCH(4);break;}
    #undef Q4_LAUNCH
    return cudaGetLastError();
}
